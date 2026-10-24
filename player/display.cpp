#include "display.h"

#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unistd.h>

bool Display::init(SDL_Window* win) {
    win_ = win;
    Uint32 n_ext = 0;
    const char* const* exts = SDL_Vulkan_GetInstanceExtensions(&n_ext);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vsrplayer";
    app.apiVersion = VK_API_VERSION_1_1;  // 1.1: device UUIDs and external memory are core
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = n_ext;
    ici.ppEnabledExtensionNames = exts;
    if (vkCreateInstance(&ici, nullptr, &inst_) != VK_SUCCESS) return fprintf(stderr, "no Vulkan\n"), false;
    if (!SDL_Vulkan_CreateSurface(win, inst_, nullptr, &surf_)) return fprintf(stderr, "%s\n", SDL_GetError()), false;

    int n_cuda = 0;
    cudaGetDeviceCount(&n_cuda);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst_, &n, nullptr);
    std::vector<VkPhysicalDevice> gpus(n);
    vkEnumeratePhysicalDevices(inst_, &n, gpus.data());
    int best_cuda = -1;
    for (VkPhysicalDevice g : gpus) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(g, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qs(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(g, &nq, qs.data());
        int fam = -1;
        for (uint32_t i = 0; i < nq && fam < 0; i++) {
            VkBool32 present = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(g, i, surf_, &present);
            if (present && (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) fam = i;  // graphics queues can blit
        }
        if (fam < 0) continue;
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
        vkGetPhysicalDeviceProperties2(g, &p2);
        int cuda = -1;
        for (int c = 0; c < n_cuda && cuda < 0; c++) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, c) == cudaSuccess && !memcmp(prop.uuid.bytes, id.deviceUUID, VK_UUID_SIZE))
                cuda = c;
        }
        if (!gpu_ || (cuda >= 0 && best_cuda < 0)) gpu_ = g, qfam_ = fam, best_cuda = cuda;
    }
    if (!gpu_) return fprintf(stderr, "no Vulkan GPU can show this window\n"), false;
    cuda_device_ = std::max(best_cuda, 0);

    uint32_t n_dev_ext = 0;
    vkEnumerateDeviceExtensionProperties(gpu_, nullptr, &n_dev_ext, nullptr);
    std::vector<VkExtensionProperties> avail(n_dev_ext);
    vkEnumerateDeviceExtensionProperties(gpu_, nullptr, &n_dev_ext, avail.data());
    auto has = [&](const char* e) {
        return std::any_of(avail.begin(), avail.end(), [&](auto& a) { return !strcmp(a.extensionName, e); });
    };
    std::vector<const char*> dev_ext{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    interop_ = best_cuda >= 0 && has(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    if (interop_) dev_ext.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);

    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfam_, qci.queueCount = 1, qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1, dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)dev_ext.size(), dci.ppEnabledExtensionNames = dev_ext.data();
    if (vkCreateDevice(gpu_, &dci, nullptr, &dev_) != VK_SUCCESS) return fprintf(stderr, "vkCreateDevice failed\n"), false;
    vkGetDeviceQueue(dev_, qfam_, 0, &queue_);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, pci.queueFamilyIndex = qfam_;
    vkCreateCommandPool(dev_, &pci, nullptr, &pool_);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool_, cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(dev_, &cai, &cmd_);
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(dev_, &fci, nullptr, &fence_);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(dev_, &sci, nullptr, &acquired_);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu_, &props);
    fprintf(stderr, "display: %s, %s\n", props.deviceName,
            interop_ ? "CUDA writes into Vulkan memory directly" : "frames copied through the CPU (no CUDA/Vulkan sharing)");
    return true;
}

uint32_t Display::memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

bool Display::alloc(int max_w, int max_h) {
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D, ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {(uint32_t)max_w, (uint32_t)max_h, 1}, ii.mipLevels = 1, ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT, ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateImage(dev_, &ii, nullptr, &img_) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev_, img_, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size, ai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(dev_, &ai, nullptr, &img_mem_) != VK_SUCCESS) return false;
    vkBindImageMemory(dev_, img_, img_mem_, 0);

    VkDeviceSize size = (VkDeviceSize)max_w * max_h * 4;
    auto make_buffer = [&](bool exported) {
        VkExternalMemoryBufferCreateInfo ebi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, exported ? &ebi : nullptr};
        bi.size = size, bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateBuffer(dev_, &bi, nullptr, &buf_) != VK_SUCCESS) return false;
        vkGetBufferMemoryRequirements(dev_, buf_, &req);
        VkExportMemoryAllocateInfo emi{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        emi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkMemoryAllocateInfo bai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, exported ? &emi : nullptr};
        bai.allocationSize = req.size;
        bai.memoryTypeIndex = memory_type(req.memoryTypeBits, exported ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                                                       : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (bai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(dev_, &bai, nullptr, &buf_mem_) != VK_SUCCESS)
            return false;
        vkBindBufferMemory(dev_, buf_, buf_mem_, 0);
        if (!exported) {
            if (vkMapMemory(dev_, buf_mem_, 0, size, 0, &host_) != VK_SUCCESS) return false;
            return cudaMalloc(&dev_ptr_, size) == cudaSuccess;
        }
        auto get_fd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev_, "vkGetMemoryFdKHR");
        VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        gi.memory = buf_mem_, gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        if (!get_fd || get_fd(dev_, &gi, &fd) != VK_SUCCESS) return false;
        cudaExternalMemoryHandleDesc hd{};
        hd.type = cudaExternalMemoryHandleTypeOpaqueFd, hd.handle.fd = fd, hd.size = req.size;
        if (cudaImportExternalMemory(&ext_, &hd) != cudaSuccess) return close(fd), false;  // CUDA owns fd on success
        cudaExternalMemoryBufferDesc bd{};
        bd.size = size;
        return cudaExternalMemoryGetMappedBuffer((void**)&dev_ptr_, ext_, &bd) == cudaSuccess;
    };
    auto drop_buffer = [&] {
        if (ext_) cudaDestroyExternalMemory(ext_), ext_ = nullptr, dev_ptr_ = nullptr;
        vkDestroyBuffer(dev_, buf_, nullptr), vkFreeMemory(dev_, buf_mem_, nullptr);
        buf_ = VK_NULL_HANDLE, buf_mem_ = VK_NULL_HANDLE;
        cudaGetLastError();  // clear the failed import's error so later CUDA calls don't report it
    };
    if (interop_ && !make_buffer(true)) {
        fprintf(stderr, "display: CUDA/Vulkan sharing failed, copying frames through the CPU\n");
        drop_buffer();
        interop_ = false;
    }
    return interop_ || make_buffer(false);
}

bool Display::make_swapchain() {
    vkDeviceWaitIdle(dev_);
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu_, surf_, &caps);
    extent_ = caps.currentExtent;
    if (extent_.width == UINT32_MAX) {  // the window system lets us choose: use the window's pixel size
        int w, h;
        SDL_GetWindowSizeInPixels(win_, &w, &h);
        extent_ = {std::clamp((uint32_t)w, caps.minImageExtent.width, caps.maxImageExtent.width),
                   std::clamp((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height)};
    }
    if (!extent_.width || !extent_.height) return false;  // minimized

    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surf_, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surf_, &n, fmts.data());
    VkSurfaceFormatKHR fmt = fmts[0];
    for (auto f : fmts)  // UNORM, not SRGB: video is already gamma-encoded, so it must be stored as-is
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) { fmt = f; break; }

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surf_;
    sci.minImageCount = caps.maxImageCount ? std::min(caps.minImageCount + 1, caps.maxImageCount) : caps.minImageCount + 1;
    sci.imageFormat = fmt.format, sci.imageColorSpace = fmt.colorSpace, sci.imageExtent = extent_;
    sci.imageArrayLayers = 1, sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    for (auto a : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR})
        if (caps.supportedCompositeAlpha & a) { sci.compositeAlpha = a; break; }
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // vsync; always supported
    sci.clipped = VK_TRUE, sci.oldSwapchain = swap_;
    VkSwapchainKHR fresh;
    if (vkCreateSwapchainKHR(dev_, &sci, nullptr, &fresh) != VK_SUCCESS) return false;
    vkDestroySwapchainKHR(dev_, swap_, nullptr);
    swap_ = fresh;

    vkGetSwapchainImagesKHR(dev_, swap_, &n, nullptr);
    images_.resize(n);
    vkGetSwapchainImagesKHR(dev_, swap_, &n, images_.data());
    for (VkSemaphore s : done_) vkDestroySemaphore(dev_, s, nullptr);
    done_.resize(n);
    VkSemaphoreCreateInfo semi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto& s : done_) vkCreateSemaphore(dev_, &semi, nullptr, &s);
    stale_ = false;
    return true;
}

uint8_t* Display::begin_frame() {
    vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX);  // the previous frame's copy out of buf_ is done
    return dev_ptr_;
}

void Display::barrier(VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src, b.dstAccessMask = dst, b.oldLayout = from, b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img, b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    // shortcut: ALL_COMMANDS stages are coarser than needed; irrelevant next to the network's cost
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
}

bool Display::present(int w, int h, cudaStream_t s) {
    if (host_) cudaMemcpyAsync(host_, dev_ptr_, (size_t)w * h * 4, cudaMemcpyDeviceToHost, s);
    // shortcut: a CPU wait instead of exported semaphores; costs a little latency, not throughput
    cudaStreamSynchronize(s);
    if (stale_ && !make_swapchain()) return false;
    uint32_t idx;
    VkResult r = vkAcquireNextImageKHR(dev_, swap_, UINT64_MAX, acquired_, VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) return stale_ = true, false;
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) return false;
    vkResetFences(dev_, 1, &fence_);

    VkImage target = images_[idx];
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(cmd_, 0);
    vkBeginCommandBuffer(cmd_, &bi);
    barrier(img_, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy copy{};
    copy.bufferRowLength = w;
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {(uint32_t)w, (uint32_t)h, 1};
    vkCmdCopyBufferToImage(cmd_, buf_, img_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    barrier(img_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    barrier(target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkClearColorValue black{};
    VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd_, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
    barrier(target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    float fit = std::min(extent_.width / (float)w, extent_.height / (float)h);
    int dw = std::max(1, (int)(w * fit)), dh = std::max(1, (int)(h * fit));
    int x0 = (extent_.width - dw) / 2, y0 = (extent_.height - dh) / 2;
    VkImageBlit blit{};
    blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {w, h, 1};
    blit.dstOffsets[0] = {x0, y0, 0};
    blit.dstOffsets[1] = {x0 + dw, y0 + dh, 1};
    vkCmdBlitImage(cmd_, img_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                   &blit, VK_FILTER_LINEAR);
    barrier(target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
    vkEndCommandBuffer(cmd_);

    VkPipelineStageFlags wait = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1, si.pWaitSemaphores = &acquired_, si.pWaitDstStageMask = &wait;
    si.commandBufferCount = 1, si.pCommandBuffers = &cmd_;
    si.signalSemaphoreCount = 1, si.pSignalSemaphores = &done_[idx];
    vkQueueSubmit(queue_, 1, &si, fence_);
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1, pi.pWaitSemaphores = &done_[idx];
    pi.swapchainCount = 1, pi.pSwapchains = &swap_, pi.pImageIndices = &idx;
    r = vkQueuePresentKHR(queue_, &pi);
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) stale_ = true;
    return true;
}

Display::~Display() {
    if (dev_) {
        vkDeviceWaitIdle(dev_);
        cudaFree(dev_ptr_);  // mapped external buffers are freed with cudaFree too
        if (ext_) cudaDestroyExternalMemory(ext_);
        vkDestroyBuffer(dev_, buf_, nullptr), vkFreeMemory(dev_, buf_mem_, nullptr);
        vkDestroyImage(dev_, img_, nullptr), vkFreeMemory(dev_, img_mem_, nullptr);
        for (VkSemaphore s : done_) vkDestroySemaphore(dev_, s, nullptr);
        vkDestroySemaphore(dev_, acquired_, nullptr);
        vkDestroyFence(dev_, fence_, nullptr);
        vkDestroyCommandPool(dev_, pool_, nullptr);
        vkDestroySwapchainKHR(dev_, swap_, nullptr);
        vkDestroyDevice(dev_, nullptr);
    }
    if (inst_) {
        if (surf_) SDL_Vulkan_DestroySurface(inst_, surf_, nullptr);
        vkDestroyInstance(inst_, nullptr);
    }
}
