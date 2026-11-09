//! ghostframe Vulkan layer.
//!
//! Sits between a game and the driver, intercepts every present and inserts
//! a generated frame between consecutive real ones. Works for native Vulkan
//! games and, on Linux, for DirectX games running through DXVK or
//! VKD3D-Proton (Steam Play / Proton), because those also present through
//! Vulkan.
//!
//! Enable per game with `ENABLE_GHOSTFRAME=1` (the installer registers the
//! layer as an implicit layer that stays off unless that variable is set).

#![allow(clippy::missing_safety_doc)]

mod fg;
mod session;

use ash::vk;
use ash::vk::Handle;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::sync::{Arc, Mutex};

// ---- loader <-> layer interface (vk_layer.h), not part of ash ------------

const LAYER_LINK_INFO: i32 = 0;
const LOADER_DATA_CALLBACK: i32 = 1;

#[repr(C)]
struct LayerInstanceLink {
    next: *mut LayerInstanceLink,
    next_gipa: vk::PFN_vkGetInstanceProcAddr,
    next_gpdpa: *const c_void,
}

#[repr(C)]
struct LayerInstanceCreateInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    function: i32,
    u: [*mut c_void; 2],
}

#[repr(C)]
struct LayerDeviceLink {
    next: *mut LayerDeviceLink,
    next_gipa: vk::PFN_vkGetInstanceProcAddr,
    next_gdpa: vk::PFN_vkGetDeviceProcAddr,
}

#[repr(C)]
struct LayerDeviceCreateInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    function: i32,
    u: *mut c_void,
}

pub type PfnSetDeviceLoaderData = unsafe extern "system" fn(vk::Device, *mut c_void) -> vk::Result;

#[repr(C)]
pub struct NegotiateLayerInterface {
    s_type: i32,
    p_next: *mut c_void,
    loader_layer_interface_version: u32,
    pfn_get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pfn_get_device_proc_addr: Option<vk::PFN_vkGetDeviceProcAddr>,
    pfn_get_physical_device_proc_addr: *const c_void,
}

// ---- per-object state ----------------------------------------------------

pub struct InstanceData {
    pub instance: vk::Instance,
    pub next_gipa: vk::PFN_vkGetInstanceProcAddr,
    pub fp: ash::InstanceFnV1_0,
    pub surface: ash::khr::surface::InstanceFn,
}

pub struct DeviceData {
    pub device: vk::Device,
    pub physical: vk::PhysicalDevice,
    pub next_gdpa: vk::PFN_vkGetDeviceProcAddr,
    pub fp: ash::DeviceFnV1_0,
    pub swap: ash::khr::swapchain::DeviceFn,
    pub set_loader_data: Option<PfnSetDeviceLoaderData>,
    pub mem: vk::PhysicalDeviceMemoryProperties,
    pub families: Vec<vk::QueueFamilyProperties>,
    /// R32_SFLOAT can be linearly filtered (fast matching path).
    pub float_filter: bool,
    pub queues: Mutex<HashMap<u64, u32>>,
}

struct SwapchainData {
    device_key: usize,
    supported: bool,
    format: vk::Format,
    extent: vk::Extent2D,
    fg: Option<fg::FrameGen>,
    failed: bool,
}

static INSTANCES: Mutex<Option<HashMap<usize, Arc<InstanceData>>>> = Mutex::new(None);
static DEVICES: Mutex<Option<HashMap<usize, Arc<DeviceData>>>> = Mutex::new(None);
static SWAPCHAINS: Mutex<Option<HashMap<u64, SwapchainData>>> = Mutex::new(None);
/// This process's entry in the tray (outer None: not created yet).
static SESSION: Mutex<Option<Option<session::Session>>> = Mutex::new(None);

/// Whether frame generation is switched on for this game right now.
fn generation_on() -> bool {
    let mut s = SESSION.lock().unwrap();
    s.get_or_insert_with(session::Session::new).as_ref().is_some_and(|s| s.enabled)
}

fn report_frame(generated: bool, extent: vk::Extent2D) {
    if let Some(Some(s)) = SESSION.lock().unwrap().as_mut() {
        s.frame(generated as u64, extent.width, extent.height);
    }
}

/// Loader dispatch key: the first pointer stored in every dispatchable handle.
unsafe fn key<H: Handle>(h: H) -> usize {
    *(h.as_raw() as *const usize)
}

fn instance_data(k: usize) -> Option<Arc<InstanceData>> {
    INSTANCES.lock().unwrap().as_ref()?.get(&k).cloned()
}

fn device_data(k: usize) -> Option<Arc<DeviceData>> {
    DEVICES.lock().unwrap().as_ref()?.get(&k).cloned()
}

pub fn log(msg: &str) {
    use std::io::Write;
    match std::env::var_os("GHOSTFRAME_LOG") {
        Some(path) if !path.is_empty() => {
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "ghostframe: {msg}");
            }
        }
        _ => eprintln!("ghostframe: {msg}"),
    }
}

unsafe fn load_fn<T>(f: vk::PFN_vkVoidFunction) -> Option<T> {
    f.map(|p| std::mem::transmute_copy::<unsafe extern "system" fn(), T>(&p))
}

// ---- instance ------------------------------------------------------------

unsafe extern "system" fn create_instance(
    ci: *const vk::InstanceCreateInfo<'_>,
    alloc: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Instance,
) -> vk::Result {
    let mut p = (*ci).p_next as *mut LayerInstanceCreateInfo;
    while !p.is_null() && !((*p).s_type == vk::StructureType::LOADER_INSTANCE_CREATE_INFO && (*p).function == LAYER_LINK_INFO) {
        p = (*p).p_next as *mut LayerInstanceCreateInfo;
    }
    if p.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let link = (*p).u[0] as *mut LayerInstanceLink;
    let next_gipa = (*link).next_gipa;
    (*p).u[0] = (*link).next as *mut c_void;

    let Some(create): Option<vk::PFN_vkCreateInstance> = load_fn(next_gipa(vk::Instance::null(), c"vkCreateInstance".as_ptr())) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let r = create(ci, alloc, out);
    if r != vk::Result::SUCCESS {
        return r;
    }
    let instance = *out;
    let get = |n: &CStr| next_gipa(instance, n.as_ptr()).map_or(std::ptr::null(), |f| f as *const c_void);
    let data = InstanceData {
        instance,
        next_gipa,
        fp: ash::InstanceFnV1_0::load(get),
        surface: ash::khr::surface::InstanceFn::load(get),
    };
    INSTANCES.lock().unwrap().get_or_insert_with(HashMap::new).insert(key(instance), Arc::new(data));
    log(&format!("loaded (v{})", env!("CARGO_PKG_VERSION")));
    r
}

unsafe extern "system" fn destroy_instance(instance: vk::Instance, alloc: *const vk::AllocationCallbacks<'_>) {
    let k = key(instance);
    let data = INSTANCES.lock().unwrap().as_mut().and_then(|m| m.remove(&k));
    if let Some(d) = data {
        (d.fp.destroy_instance)(instance, alloc);
    }
}

// ---- device --------------------------------------------------------------

unsafe extern "system" fn create_device(
    physical: vk::PhysicalDevice,
    ci: *const vk::DeviceCreateInfo<'_>,
    alloc: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::Device,
) -> vk::Result {
    let Some(inst) = instance_data(key(physical)) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let mut link_ci: *mut LayerDeviceCreateInfo = std::ptr::null_mut();
    let mut set_loader_data: Option<PfnSetDeviceLoaderData> = None;
    let mut p = (*ci).p_next as *mut LayerDeviceCreateInfo;
    while !p.is_null() {
        if (*p).s_type == vk::StructureType::LOADER_DEVICE_CREATE_INFO {
            if (*p).function == LAYER_LINK_INFO && link_ci.is_null() {
                link_ci = p;
            } else if (*p).function == LOADER_DATA_CALLBACK {
                set_loader_data = Some(std::mem::transmute::<*mut c_void, PfnSetDeviceLoaderData>((*p).u));
            }
        }
        p = (*p).p_next as *mut LayerDeviceCreateInfo;
    }
    if link_ci.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    let link = (*link_ci).u as *mut LayerDeviceLink;
    let next_gipa = (*link).next_gipa;
    let next_gdpa = (*link).next_gdpa;
    (*link_ci).u = (*link).next as *mut c_void;

    let Some(create): Option<vk::PFN_vkCreateDevice> = load_fn(next_gipa(inst.instance, c"vkCreateDevice".as_ptr())) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let r = create(physical, ci, alloc, out);
    if r != vk::Result::SUCCESS {
        return r;
    }
    let device = *out;
    if set_loader_data.is_none() {
        log("loader gave no device-loader-data callback; frame generation off for this device");
    }
    let get = |n: &CStr| next_gdpa(device, n.as_ptr()).map_or(std::ptr::null(), |f| f as *const c_void);

    let mut mem = vk::PhysicalDeviceMemoryProperties::default();
    (inst.fp.get_physical_device_memory_properties)(physical, &mut mem);
    let mut n = 0u32;
    (inst.fp.get_physical_device_queue_family_properties)(physical, &mut n, std::ptr::null_mut());
    let mut families = vec![vk::QueueFamilyProperties::default(); n as usize];
    (inst.fp.get_physical_device_queue_family_properties)(physical, &mut n, families.as_mut_ptr());
    let mut fprops = vk::FormatProperties::default();
    (inst.fp.get_physical_device_format_properties)(physical, vk::Format::R32_SFLOAT, &mut fprops);
    let float_filter = fprops.optimal_tiling_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);

    let data = DeviceData {
        device,
        physical,
        next_gdpa,
        fp: ash::DeviceFnV1_0::load(get),
        swap: ash::khr::swapchain::DeviceFn::load(get),
        set_loader_data,
        mem,
        families,
        float_filter,
        queues: Mutex::new(HashMap::new()),
    };
    DEVICES.lock().unwrap().get_or_insert_with(HashMap::new).insert(key(device), Arc::new(data));
    r
}

unsafe extern "system" fn destroy_device(device: vk::Device, alloc: *const vk::AllocationCallbacks<'_>) {
    let k = key(device);
    // Free frame generators that belong to this device first.
    let gens: Vec<fg::FrameGen> = {
        let mut sw = SWAPCHAINS.lock().unwrap();
        let mut out = Vec::new();
        if let Some(m) = sw.as_mut() {
            for s in m.values_mut().filter(|s| s.device_key == k) {
                if let Some(g) = s.fg.take() {
                    out.push(g);
                }
            }
            m.retain(|_, s| s.device_key != k);
        }
        out
    };
    for g in gens {
        g.destroy();
    }
    let data = DEVICES.lock().unwrap().as_mut().and_then(|m| m.remove(&k));
    if let Some(d) = data {
        (d.fp.destroy_device)(device, alloc);
    }
}

unsafe extern "system" fn get_device_queue(device: vk::Device, family: u32, index: u32, out: *mut vk::Queue) {
    if let Some(d) = device_data(key(device)) {
        (d.fp.get_device_queue)(device, family, index, out);
        d.queues.lock().unwrap().insert((*out).as_raw(), family);
    }
}

unsafe extern "system" fn get_device_queue2(device: vk::Device, info: *const vk::DeviceQueueInfo2<'_>, out: *mut vk::Queue) {
    if let Some(d) = device_data(key(device)) {
        if let Some(f) = load_fn::<vk::PFN_vkGetDeviceQueue2>((d.next_gdpa)(device, c"vkGetDeviceQueue2".as_ptr())) {
            f(device, info, out);
            d.queues.lock().unwrap().insert((*out).as_raw(), (*info).queue_family_index);
        }
    }
}

// ---- swapchain -----------------------------------------------------------

fn supported_format(f: vk::Format) -> bool {
    matches!(
        f,
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB
    )
}

unsafe extern "system" fn create_swapchain(
    device: vk::Device,
    ci: *const vk::SwapchainCreateInfoKHR<'_>,
    alloc: *const vk::AllocationCallbacks<'_>,
    out: *mut vk::SwapchainKHR,
) -> vk::Result {
    let Some(d) = device_data(key(device)) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let inst = instance_data(key(d.physical));
    let mut info = *ci;
    let mut supported = supported_format(info.image_format) && std::env::var("GHOSTFRAME_MODE").map_or(true, |m| m != "off");
    if supported {
        let mut caps = vk::SurfaceCapabilitiesKHR::default();
        let ok = inst.is_some_and(|i| {
            (i.surface.get_physical_device_surface_capabilities_khr)(d.physical, info.surface, &mut caps) == vk::Result::SUCCESS
        });
        let need = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if ok && caps.supported_usage_flags.contains(need) {
            info.image_usage |= need;
            // One extra image to hold the generated frame.
            info.min_image_count += 1;
            if caps.max_image_count != 0 {
                info.min_image_count = info.min_image_count.min(caps.max_image_count);
            }
        } else {
            supported = false;
        }
    }
    let r = (d.swap.create_swapchain_khr)(device, &info, alloc, out);
    if r != vk::Result::SUCCESS && supported {
        // Retry exactly as the game asked.
        return (d.swap.create_swapchain_khr)(device, ci, alloc, out);
    }
    if r == vk::Result::SUCCESS {
        if !supported {
            log(&format!("swapchain format {:?} not supported, passing through", info.image_format));
        }
        SWAPCHAINS.lock().unwrap().get_or_insert_with(HashMap::new).insert(
            (*out).as_raw(),
            SwapchainData { device_key: key(device), supported, format: info.image_format, extent: info.image_extent, fg: None, failed: false },
        );
    }
    r
}

unsafe extern "system" fn destroy_swapchain(device: vk::Device, swapchain: vk::SwapchainKHR, alloc: *const vk::AllocationCallbacks<'_>) {
    let Some(d) = device_data(key(device)) else { return };
    let data = SWAPCHAINS.lock().unwrap().as_mut().and_then(|m| m.remove(&swapchain.as_raw()));
    if let Some(g) = data.and_then(|s| s.fg) {
        g.destroy();
    }
    (d.swap.destroy_swapchain_khr)(device, swapchain, alloc);
}

unsafe extern "system" fn queue_present(queue: vk::Queue, info: *const vk::PresentInfoKHR<'_>) -> vk::Result {
    let Some(d) = device_data(key(queue)) else {
        return vk::Result::ERROR_DEVICE_LOST;
    };
    let pass = |d: &DeviceData| (d.swap.queue_present_khr)(queue, info);
    if (*info).swapchain_count != 1 {
        return pass(&d);
    }
    let sc_handle = *(*info).p_swapchains;
    let family = d.queues.lock().unwrap().get(&queue.as_raw()).copied();
    let mut sw = SWAPCHAINS.lock().unwrap();
    let Some(sc) = sw.as_mut().and_then(|m| m.get_mut(&sc_handle.as_raw())) else {
        drop(sw);
        return pass(&d);
    };
    let extent = sc.extent;
    // Not switched on in the tray (the default): the game runs untouched,
    // and the engine is not even created until the first click.
    if !sc.supported || sc.failed || !generation_on() {
        if let Some(g) = sc.fg.as_mut() {
            g.pause();
        }
        drop(sw);
        let r = pass(&d);
        report_frame(false, extent);
        return r;
    }
    if sc.fg.is_none() && d.set_loader_data.is_none() {
        sc.failed = true;
        drop(sw);
        return pass(&d);
    }
    if sc.fg.is_none() {
        let fam = family.unwrap_or(0);
        let compute = d.families.get(fam as usize).is_some_and(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE));
        if !compute {
            log("present queue has no compute support, passing through");
            sc.failed = true;
            drop(sw);
            return pass(&d);
        }
        match fg::FrameGen::new(d.clone(), queue, fam, sc_handle, sc.format, sc.extent) {
            Ok(g) => {
                log(&format!(
                    "frame generation on: {}x{} {:?}, {} swapchain images",
                    sc.extent.width,
                    sc.extent.height,
                    sc.format,
                    g.image_count()
                ));
                sc.fg = Some(g);
            }
            Err(e) => {
                log(&format!("could not start frame generation ({e}), passing through"));
                sc.failed = true;
                drop(sw);
                return pass(&d);
            }
        }
    }
    let (r, generated) = sc.fg.as_mut().unwrap().present(queue, &*info);
    drop(sw);
    report_frame(generated, extent);
    r
}

// ---- dispatch ------------------------------------------------------------

unsafe fn intercept(name: &CStr) -> vk::PFN_vkVoidFunction {
    macro_rules! f {
        ($f:expr) => {
            Some(std::mem::transmute::<*const (), unsafe extern "system" fn()>($f as *const ()))
        };
    }
    match name.to_bytes() {
        b"vkGetInstanceProcAddr" => f!(vkGetInstanceProcAddr),
        b"vkGetDeviceProcAddr" => f!(vkGetDeviceProcAddr),
        b"vkCreateInstance" => f!(create_instance),
        b"vkDestroyInstance" => f!(destroy_instance),
        b"vkCreateDevice" => f!(create_device),
        b"vkDestroyDevice" => f!(destroy_device),
        b"vkGetDeviceQueue" => f!(get_device_queue),
        b"vkGetDeviceQueue2" => f!(get_device_queue2),
        b"vkCreateSwapchainKHR" => f!(create_swapchain),
        b"vkDestroySwapchainKHR" => f!(destroy_swapchain),
        b"vkQueuePresentKHR" => f!(queue_present),
        _ => None,
    }
}

#[no_mangle]
pub unsafe extern "system" fn vkGetInstanceProcAddr(instance: vk::Instance, name: *const c_char) -> vk::PFN_vkVoidFunction {
    let n = CStr::from_ptr(name);
    if let Some(f) = intercept(n) {
        return Some(f);
    }
    if instance.is_null() {
        return None;
    }
    let data = instance_data(key(instance))?;
    (data.next_gipa)(instance, name)
}

#[no_mangle]
pub unsafe extern "system" fn vkGetDeviceProcAddr(device: vk::Device, name: *const c_char) -> vk::PFN_vkVoidFunction {
    let n = CStr::from_ptr(name);
    let data = device_data(key(device))?;
    if let Some(f) = intercept(n) {
        // Only hand out device-level hooks we actually implement.
        if !matches!(n.to_bytes(), b"vkCreateInstance" | b"vkDestroyInstance" | b"vkCreateDevice") {
            return Some(f);
        }
    }
    (data.next_gdpa)(device, name)
}

#[no_mangle]
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(p: *mut NegotiateLayerInterface) -> vk::Result {
    if p.is_null() || (*p).loader_layer_interface_version < 2 {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    (*p).loader_layer_interface_version = 2;
    (*p).pfn_get_instance_proc_addr = Some(vkGetInstanceProcAddr);
    (*p).pfn_get_device_proc_addr = Some(vkGetDeviceProcAddr);
    (*p).pfn_get_physical_device_proc_addr = std::ptr::null();
    vk::Result::SUCCESS
}
