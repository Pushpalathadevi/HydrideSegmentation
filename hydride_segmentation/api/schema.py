"""Pydantic models for request validation."""
from __future__ import annotations

from typing import Literal, Tuple
from pydantic import BaseModel, field_validator


class SegmentParams(BaseModel):
    model: Literal["conventional", "ml"] = "conventional"
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] | str = (8, 8)
    adaptive_window: int = 31
    adaptive_offset: int = 2
    morph_kernel: int = 3
    morph_iters: int = 1

    @field_validator("clahe_clip_limit")
    @classmethod
    def _clip_limit(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("clahe_clip_limit must be > 0")
        return v

    @field_validator("clahe_tile_grid", mode="before")
    @classmethod
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

    @field_validator("adaptive_window")
    @classmethod
    def _adaptive_window(cls, v: int) -> int:
        if v < 3 or v % 2 == 0:
            raise ValueError("adaptive_window must be odd and >= 3")
        return v

    @field_validator("morph_kernel")
    @classmethod
    def _morph_kernel(cls, v: int) -> int:
        if v < 1 or v % 2 == 0:
            raise ValueError("morph_kernel must be odd and >= 1")
        return v

    @field_validator("morph_iters")
    @classmethod
    def _morph_iters(cls, v: int) -> int:
        if v < 0:
            raise ValueError("morph_iters must be >= 0")
        return v
