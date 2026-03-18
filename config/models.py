from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    name: str
    mic_type: str
    column_range: range
    thd_column_index: int | None = None

    @property
    def has_thd(self) -> bool:
        return self.thd_column_index is not None


@dataclass(frozen=True, slots=True)
class ProductSpec:
    model_name: str
    pn_codes: tuple[str, ...]
    channels: tuple[ChannelSpec, ...]

    def matches_pn(self, pn_code: str) -> bool:
        return pn_code in self.pn_codes


class ProductCatalog:
    def __init__(self, products: tuple[ProductSpec, ...]):
        self._products = products
        self._products_by_name = {product.model_name: product for product in products}

    def model_names(self) -> tuple[str, ...]:
        return tuple(self._products_by_name.keys())

    def get(self, model_name: str) -> ProductSpec:
        return self._products_by_name[model_name]

    def detect_by_pn(self, pn_code: str) -> ProductSpec | None:
        for product in self._products:
            if product.matches_pn(pn_code):
                return product
        return None


@dataclass(frozen=True, slots=True)
class LimitPolicy:
    check_points: tuple[int, ...]
    defect_types: tuple[str, ...]
    control_limit_specs: tuple[tuple[object, ...], ...]
    cpk_limits: dict[str, tuple[float, float]]
    no_signal_limits: dict[str, float]
    thd_limits: dict[str, float]

    def cpk_limit_for(self, mic_type: str) -> tuple[float, float]:
        return self.cpk_limits[mic_type]

    def no_signal_limit_for(self, mic_type: str) -> float:
        return self.no_signal_limits[mic_type]

    def thd_limit_for(self, mic_type: str) -> float:
        return self.thd_limits[mic_type]
