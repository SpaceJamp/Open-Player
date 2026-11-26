#pragma once
// TensorRT for RTX, seen from the rest of the player. Its SDK's headers share names and the nvinfer1 namespace with
// the full TensorRT, so only trt_rtx.cpp includes them, once per supported version (HAVE_TRT_RTX_1_<minor>), each
// copy under its own renamed namespace; everything here is plain C++ the player can include anywhere.

#ifdef HAVE_TRT_RTX_1_8
int trt_rtx_header_version_rtx8();  // NV_TENSORRT_VERSION of the headers that copy was built with
#endif
#ifdef HAVE_TRT_RTX_1_7
int trt_rtx_header_version_rtx7();
#endif

// The header version a build's copy for this minor version was compiled with; 0: not compiled in.
inline int trt_rtx_header_version(int minor) {
#ifdef HAVE_TRT_RTX_1_8
    if (minor == 8) return trt_rtx_header_version_rtx8();
#endif
#ifdef HAVE_TRT_RTX_1_7
    if (minor == 7) return trt_rtx_header_version_rtx7();
#endif
    (void)minor;
    return 0;
}
