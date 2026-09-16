add_mlir_dialect_library(DigitalCimIR
        DigitalCimBase.cpp
        DigitalCimOps.cpp
        DigitalCimTypes.cpp

    DEPENDS
        DigitalCimIncGen

    LINK_LIBS PUBLIC
        MLIRIR
        MLIRParser
        MLIRSideEffectInterfaces
)
