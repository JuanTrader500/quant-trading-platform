"""
features/ohlc_validation.py
----------------------------
RF20: rechaza y notifica cuando los valores OHLC ingresados
manualmente en modo Testing sean matemáticamente inválidos.
"""


class InvalidOHLCError(ValueError):
    """OHLC matemáticamente inconsistente (Low > High, Open/Close fuera
    de [Low, High], o valores no positivos)."""


def validate_ohlc(open_: float, high: float, low: float, close: float) -> None:
    errors: list[str] = []

    if any(v is None for v in (open_, high, low, close)):
        raise InvalidOHLCError("Open, High, Low y Close son obligatorios.")

    if any(v <= 0 for v in (open_, high, low, close)):
        errors.append("Todos los valores OHLC deben ser positivos.")

    if low > high:
        errors.append(f"Low ({low}) no puede ser mayor que High ({high}).")

    if not (low <= open_ <= high):
        errors.append(f"Open ({open_}) debe estar dentro del rango [Low, High] = [{low}, {high}].")

    if not (low <= close <= high):
        errors.append(f"Close ({close}) debe estar dentro del rango [Low, High] = [{low}, {high}].")

    if errors:
        raise InvalidOHLCError(" ".join(errors))
