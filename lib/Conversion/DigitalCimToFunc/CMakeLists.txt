add_mlir_conversion_library(MLIRDigitalCimToFunc
    DigitalCimToFunc.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/digitalcim/Conversion/DigitalCimToFunc

    DEPENDS
    DigitalCimConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    DigitalCimIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRTransformUtils
    MLIRReconcileUnrealizedCasts
)
