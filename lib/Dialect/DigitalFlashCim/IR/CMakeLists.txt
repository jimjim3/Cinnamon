add_mlir_dialect_library(DigitalFlashCimIR
        DigitalFlashCimBase.cpp
        DigitalFlashCimOps.cpp
        DigitalFlashCimTypes.cpp

    DEPENDS
        DigitalFlashCimIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
)
