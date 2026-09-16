add_library(
    DigitalCimDialectRuntime
    SHARED
    runtime_c_interface.cpp
    executor.cpp
)

install(TARGETS DigitalCimDialectRuntime LIBRARY DESTINATION lib)
