#include <d3d12.h>
#include "d3dx12.h"
#include <dxgi1_3.h>
#include <dxgi1_4.h>
#include <dxgi1_6.h>
#include <Windows.h>
#include <comdef.h>
#include <string>

// D3D12 error checking macro
#define D3D_CHECK(call) do { \
    HRESULT hr = call; \
    if (FAILED(hr)) { \
        std::cerr << "D3D12 error in " << __FILE__ << " line " << __LINE__ << ":\n"; \
        std::cerr << "Error code: 0x" << std::hex << hr << std::dec << "\n"; \
        std::cerr << "Error description: " << getD3D12ErrorString(hr) << std::endl; \
        return false; \
    } \
} while(0)

inline std::string getD3D12ErrorString(HRESULT hr) {
    switch (hr) {
        case E_INVALIDARG:                 return "Invalid argument";
        case E_OUTOFMEMORY:                return "Out of memory";
        case E_NOTIMPL:                    return "Not implemented";
        case E_FAIL:                       return "Unspecified failure";
        case E_HANDLE:                     return "Invalid handle";
        case DXGI_ERROR_INVALID_CALL:      return "Invalid DXGI call";
        case DXGI_ERROR_DEVICE_REMOVED:    return "DXGI device removed";
        case DXGI_ERROR_DEVICE_RESET:      return "DXGI device reset";
        case DXGI_ERROR_UNSUPPORTED:       return "Unsupported DXGI operation";
        case DXGI_ERROR_NOT_FOUND:         return "DXGI resource not found";
        case D3D12_ERROR_ADAPTER_NOT_FOUND: return "D3D12 adapter not found";
        case D3D12_ERROR_DRIVER_VERSION_MISMATCH: return "D3D12 driver version mismatch";
        default:
            if (HRESULT_FACILITY(hr) == FACILITY_WINDOWS) {
                char message[256] = {};
                FormatMessageA(
                    FORMAT_MESSAGE_FROM_SYSTEM,
                    nullptr,
                    hr,
                    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                    message,
                    sizeof(message),
                    nullptr
                );
                return message;
            }
            return "Unknown error code";
    }
}