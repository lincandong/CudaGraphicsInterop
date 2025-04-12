#include "HandleHelper.h"

#ifdef _WIN32
bool OpenWindowsFileAndReadContent(const std::string& ResName, void* dst, size_t size) 
{
    std::wstring wResName(ResName.begin(), ResName.end());

    HANDLE namedMapping = OpenFileMappingW(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        wResName.c_str());
    if (namedMapping == NULL) {
        std::cerr << "Failed to open file: " << ResName << std::endl;
        return false;
    }

    // read handle value from the file
    // Map the file into memory and read the handle value
    LPVOID pView = MapViewOfFile(
        namedMapping,            // Handle
        FILE_MAP_READ,           // Access mode
        0, 0,                    // Offset
        sizeof(HANDLE));         // Size
    
    if (!pView) {
        CloseHandle(namedMapping);
        std::cerr << "Failed to map file: " << ResName << std::endl;
        return false;
    }
    
    // Store the handle value in the mapping
    memcpy(dst, pView, size);
    UnmapViewOfFile(pView);
    CloseHandle(namedMapping);
    
    return true;
}
#else
// ShmName should starts with '/'
bool OpenLinuxFDAndReadContent(const std::string& ShmName, void* dst, size_t size)
{    
    if (ShmName[0] != '/')
    {
        std::cerr << "ShmName should starts with a /" << std::endl;
        return false;
    }

    int shmfd = shm_open(ShmName.c_str(), O_RDONLY, 0);
    if (shmfd < 0) {
        std::cerr << "Failed to open shared memory." << std::endl;
        return false;
    }
    
    // Map the shared memory into address space
    void *ptr = mmap(NULL, size, PROT_READ, MAP_SHARED, shmfd, 0);
    if (ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory." << std::endl;
        close(shmfd);
        return false;
    }
    
    // Store the fd in shared memory
    memcpy(dst, ptr, size);
    munmap(ptr, size);

    close(shmfd);

    return true;
}


// Wrapper for pidfd_open syscall
int pidfd_open(pid_t pid, unsigned int flags) {
    return syscall(SYS_pidfd_open, pid, flags);
}

// Wrapper for pidfd_getfd syscall
int pidfd_getfd(int pidfd, int targetfd, unsigned int flags) {
    return syscall(SYS_pidfd_getfd, pidfd, targetfd, flags);
}

int DuplicateLinuxFD(pid_t target_pid, int target_fd) {
    // Open a pidfd for the target process
    int pidfd = pidfd_open(target_pid, 0);
    if (pidfd == -1) {
        perror("pidfd_open failed");
        return -1;
    }
    
    // Get the file descriptor from the target process
    int new_fd = pidfd_getfd(pidfd, target_fd, 0);
    if (new_fd == -1) {
        perror("pidfd_getfd failed");
        close(pidfd);
        return -1;
    }
    
    // Clean up
    close(new_fd);
    close(pidfd);
    
    return new_fd;
}

int receive_fd(int socket_fd) {
    struct msghdr msg = {0};
    struct cmsghdr *cmsg;
    char buf[CMSG_SPACE(sizeof(int))];
    char dummy_byte;
    int received_fd = -1;
    
    // We need iovec to receive at least one byte of data
    struct iovec io = { .iov_base = &dummy_byte, .iov_len = 1 };
    
    msg.msg_iov = &io;
    msg.msg_iovlen = 1;
    msg.msg_control = buf;
    msg.msg_controllen = sizeof(buf);
    
    int rv = recvmsg(socket_fd, &msg, 0);
    if (rv <= 0) {
        perror("recvmsg failed");  // This will show the specific error
        fprintf(stderr, "Failed to receive data\n");
        return -1; 
    }
    
    cmsg = CMSG_FIRSTHDR(&msg);
    if (cmsg == NULL) {
        fprintf(stderr, "No control message received\n");
        return -1;
    }
    
    // Verify this is the right type of control message
    if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
        fprintf(stderr, "Invalid control message received\n");
        return -1;
    }
    
    // Extract the file descriptor
    memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));
    
    return received_fd;
}


int connect_to_server(const char *socket_path) {
    int fd;
    struct sockaddr_un addr;
    
    // Create the socket
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd == -1) {
        perror("socket error");
        return -1;
    }
    
    // Configure socket address
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
    
    // Connect to the server
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
        perror("connect error");
        close(fd);
        return -1;
    }
    
    return fd; // This is your socket_fd
}

int ReceiveLinuxFD(const std::string& sharedName)
{
    // Prefix the name with '/' as required by POSIX
    char socket_path[256] = {0};
    snprintf(socket_path, sizeof(socket_path), "/tmp/%s", sharedName.c_str());
    
    // First, connect to the sender
    std::cout << "connecting to socket at path: " << socket_path << std::endl;
    int socket_fd = connect_to_server(socket_path);
    if (socket_fd < 0) {
        fprintf(stderr, "Failed to connect to server\n");
        return -1;
    }

    // Receive the FD
    std::cout << "receiving data from socket fd: " << socket_fd << std::endl;
    int received_fd = receive_fd(socket_fd);
    close(socket_fd);

    return received_fd;
}
#endif


int GetVKMemHandle(VkDevice device, VkDeviceMemory memory) {
#ifdef _WIN32
    HANDLE handle = 0;

    VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
    vkMemoryGetWin32HandleInfoKHR.sType =
        VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
    vkMemoryGetWin32HandleInfoKHR.memory = memory;
    vkMemoryGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;

    PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR;
    fpGetMemoryWin32HandleKHR =
        (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
            device, "vkGetMemoryWin32HandleKHR");
    if (!fpGetMemoryWin32HandleKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
    }
    if (fpGetMemoryWin32HandleKHR(device, &vkMemoryGetWin32HandleInfoKHR,
                                &handle) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
    }
    return (int)handle;
#else
    int fd = -1;

    VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
    vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    vkMemoryGetFdInfoKHR.pNext = NULL;
    vkMemoryGetFdInfoKHR.memory = memory;
    vkMemoryGetFdInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
    fpGetMemoryFdKHR =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!fpGetMemoryFdKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
    }
    if (fpGetMemoryFdKHR(device, &vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
    }
    return fd;
#endif /* _WIN64 */
}