include(CMakeParseArguments)

# This little macro lets you set any Xcode specific property
macro (set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE)
    set_property (TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY} ${XCODE_VALUE})
endmacro ()

macro(add_project target)

    # parse the arguments
    cmake_parse_arguments(THIS "STATIC" "" "SOURCES" ${ARGN})
    if (NOT "${THIS_UNPARSED_ARGUMENTS}" STREQUAL "")
        message(FATAL_ERROR "Extra unparsed arguments when calling add_project: ${THIS_UNPARSED_ARGUMENTS}")
    endif()

    set(
        TARGET_PROJECT
        ${target}
    )

    # add sources
    add_executable(${TARGET_PROJECT}
        ${COMMON_SOURCES}
        ${THIS_SOURCES}
    )
    # include search path
    target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})
    # link external libs
    target_link_libraries(${TARGET_PROJECT}
        ${ALL_LIBS}
    )
    target_compile_definitions(${TARGET_PROJECT} PRIVATE ${ALL_LIBS_COMPILATION_DEFINITION})
    
    # Xcode and Visual working directories
    set_target_properties(${TARGET_PROJECT} PROPERTIES XCODE_ATTRIBUTE_CONFIGURATION_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_PROJECT}/")

    # add_custom_command(
    #     TARGET ${TARGET_PROJECT} POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${TARGET_PROJECT}${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_PROJECT}/"
    # )
endmacro()
