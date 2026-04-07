from __future__ import annotations

import io

import chardet
import pandas as pd

from config.specs import ProductCatalog
from core.domain import ProductDetection


def clean_sn(val):
    if pd.isna(val):
        return ""
    return str(val).replace('"', '').replace("'", "").replace('\t', '').strip()


def get_freq_values(cols):
    return [float(str(c).split('.')[0]) for c in cols]


class UploadedCsvReader:
    def read(self, uploaded_file):
        raw_bytes = uploaded_file.read()
        detected_encoding = chardet.detect(raw_bytes).get("encoding")
        decoded = raw_bytes.decode(detected_encoding if detected_encoding else "utf-8-sig", errors="replace")
        df = pd.read_csv(io.StringIO(decoded), low_memory=False)
        df.columns = [str(c) for c in df.columns]
        return df


class ProductDetector:
    def __init__(self, product_catalog: ProductCatalog):
        self._product_catalog = product_catalog

    def detect(self, df) -> ProductDetection:
        model_name = None
        prod_date = "Unknown"
        detected_pn = "Unknown"

        if df.shape[1] <= 3:
            return ProductDetection(model_name=model_name, prod_date=prod_date, detected_pn=detected_pn)

        for i in range(2, min(15, len(df))):
            sn_raw = clean_sn(df.iloc[i, 3])
            if "/" not in sn_raw:
                continue

            parts = sn_raw.split("/", 1)
            if len(parts) != 2:
                continue

            pn_part, sn_part = parts
            product_spec = self._product_catalog.detect_by_pn(pn_part)
            if product_spec is not None:
                model_name = product_spec.model_name
                detected_pn = pn_part

            if len(sn_part) >= 8:
                prod_date = f"20{sn_part[2:4]}/{sn_part[4:6]}/{sn_part[6:8]}"

            if model_name:
                break

        return ProductDetection(model_name=model_name, prod_date=prod_date, detected_pn=detected_pn)
