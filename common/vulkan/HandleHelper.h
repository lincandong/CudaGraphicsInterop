#pragma once

#ifndef HANDLE_HELPER_H
#define HANDLE_HELPER_H

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <windows.h>
#include <vulkan/vulkan_win32.h>
bool OpenWindowsFileAndReadContent(const std::string& ResName, void* dst, size_t size);

#else // linux
#include <vulkan/vulkan_core.h>
// #define _GNU_SOURCE
#include <fcntl.h>
#include <unistd.h>
#include <sys/un.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/mman.h>  // For shm_open
#include <sys/stat.h>  // For S_IRUSR, S_IWUSR, etc.

// Define the syscall numbers if they're not in your headers
#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif

#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif

int ReceiveLinuxFD(const std::string& sharedName);

#endif // _WIN32

int GetVKMemHandle(VkDevice device, VkDeviceMemory memory);

#endif