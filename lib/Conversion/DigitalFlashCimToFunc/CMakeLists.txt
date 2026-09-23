add_mlir_conversion_library(MLIRDigitalFlashCimToFunc
    DigitalFlashCimToFunc.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/digitalflashcim/Conversion/DigitalFlashCimToFunc

    DEPENDS
    DigitalFlashCimConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    DigitalFlashCimIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
