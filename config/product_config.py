from config.specs import ChannelSpec, ProductCatalog, ProductSpec


PRODUCT_CATALOG = ProductCatalog(
    (
        ProductSpec(
            model_name="RH",
            pn_codes=("96575N1100", "96575GJ100"),
            channels=(
                ChannelSpec("Digital MIC1", "digital", range(51, 101), 15),
                ChannelSpec("Digital MIC2", "digital", range(103, 153), 18),
                ChannelSpec("Digital MIC3", "digital", range(155, 205), 21),
            ),
        ),
        ProductSpec(
            model_name="3903(LH Ecall)",
            pn_codes=("96575N1050", "96575GJ000"),
            channels=(
                ChannelSpec("Ecall MIC (Analog)", "analog", range(6, 47), 69),
                ChannelSpec("Digital MIC1", "digital", range(107, 157), 217),
                ChannelSpec("Digital MIC2", "digital", range(159, 209), 220),
            ),
        ),
        ProductSpec(
            model_name="3203(LH non Ecall)",
            pn_codes=("96575N1000", "96575GJ010"),
            channels=(
                ChannelSpec("Digital MIC1", "digital", range(6, 56), 116),
                ChannelSpec("Digital MIC2", "digital", range(58, 108), 119),
            ),
        ),
        ProductSpec(
            model_name="LITE(LH)",
            pn_codes=("96575NR000", "96575GJ200"),
            channels=(
                ChannelSpec("Analog MIC", "analog", range(6, 47), 95),
            ),
        ),
        ProductSpec(
            model_name="LITE(RH)",
            pn_codes=("96575NR100", "96575GJ300"),
            channels=(
                ChannelSpec("Analog MIC", "analog", range(6, 47), 95),
            ),
        ),
    )
)
