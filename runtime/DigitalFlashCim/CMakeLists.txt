add_library(
    DigitalFlashCimDialectRuntime
    SHARED
    runtime_c_interface.cpp
    executor.cpp
)

install(TARGETS DigitalFlashCimDialectRuntime LIBRARY DESTINATION lib)
