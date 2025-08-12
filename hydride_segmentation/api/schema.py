"""Pydantic models for request validation."""
from __future__ import annotations

from typing import Literal, Tuple
from pydantic import BaseModel, validator


class SegmentParams(BaseModel):
    model: Literal["conventional", "ml"] = "conventional"
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] | str = (8, 8)
    adaptive_window: int = 31
    adaptive_offset: int = 2
    morph_kernel: int = 3
    morph_iters: int = 1

    @validator("clahe_clip_limit")
    def _clip_limit(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("clahe_clip_limit must be > 0")
        return v

    @validator("clahe_tile_grid", pre=True)
    def _tile_grid(cls, v):
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError("clahe_tile_grid must have two values")
            x, y = int(v[0]), int(v[1])
        elif isinstance(v, str):
            parts = v.split(",")
            if len(parts) != 2:
                raise ValueError("clahe_tile_grid must be 'X,Y'")
            x, y = int(parts[0]), int(parts[1])
        else:
            raise ValueError("clahe_tile_grid must be 'X,Y'")
        if x < 1 or y < 1:
            raise ValueError("clahe_tile_grid values must be >= 1")
        return (x, y)

    @validator("adaptive_window")
    def _adaptive_window(cls, v: int) -> int:
        if v < 3 or v % 2 == 0:
            raise ValueError("adaptive_window must be odd and >= 3")
        return v

    @validator("morph_kernel")
    def _morph_kernel(cls, v: int) -> int:
        if v < 1 or v % 2 == 0:
            raise ValueError("morph_kernel must be odd and >= 1")
        return v

    @validator("morph_iters")
    def _morph_iters(cls, v: int) -> int:
        if v < 0:
            raise ValueError("morph_iters must be >= 0")
        return v
