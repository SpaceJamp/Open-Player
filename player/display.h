#pragma once
#include <SDL3/SDL.h>
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

// Vulkan presenter. CUDA writes RGBA8 straight into a Vulkan buffer (external memory), which is copied to an
// image and blitted, letterboxed and filtered, onto the swapchain. No shaders or render passes needed.
// Where CUDA and Vulkan can't share memory (e.g. WSL), frames go through a host-visible buffer instead.
class Display {
public:
    ~Display();
    bool init(SDL_Window* win);  // picks the GPU; cuda_device() is then the same GPU on the CUDA side
    int cuda_device() const { return cuda_device_; }
    bool zero_copy() const { return host_ == nullptr; }  // CUDA writes straight into Vulkan memory
    const char* gpu_name() const { return gpu_name_; }
    // diagnostics from the latest present(): GPU -> CPU copy, CPU -> Vulkan memory copy (fallback only), present
    double last_copy_ms = 0, last_map_ms = 0, last_present_ms = 0;
    bool alloc(int max_w, int max_h);  // frame buffer for images up to max_w x max_h
    uint8_t* begin_frame();            // waits until the GPU is done with the previous frame; write RGBA8 here
    // Shows the w x h image written since begin_frame(). False if nothing could be shown (e.g. minimized).
    bool present(int w, int h, cudaStream_t s);
    void resized() { stale_ = true; }

private:
    bool make_swapchain();
    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid = 0);
    void barrier(VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst);

    SDL_Window* win_ = nullptr;
    VkInstance inst_ = VK_NULL_HANDLE;
    VkSurfaceKHR surf_ = VK_NULL_HANDLE;
    VkPhysicalDevice gpu_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qfam_ = 0;
    VkSwapchainKHR swap_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkSemaphore> done_;  // one per swapchain image: rendering finished, safe to present
    VkSemaphore acquired_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    bool stale_ = true, interop_ = false;

    VkImage img_ = VK_NULL_HANDLE;
    VkDeviceMemory img_mem_ = VK_NULL_HANDLE;
    VkBuffer buf_ = VK_NULL_HANDLE;
    VkDeviceMemory buf_mem_ = VK_NULL_HANDLE;
    cudaExternalMemory_t ext_ = nullptr;
    uint8_t* dev_ptr_ = nullptr;  // where CUDA writes
    void* host_ = nullptr;        // fallback only: mapped Vulkan memory the frame is copied into
    void* pinned_ = nullptr;      // fallback only: CUDA-pinned staging copy on the way there
    bool registered_ = false;     // fallback only: host_ registered with CUDA, so the GPU writes it directly
    int cuda_device_ = 0;
    char gpu_name_[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE] = "";
};
