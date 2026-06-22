"use client";

import { useMemo } from "react";
import { chartCssVars, useChartStable, useYScale } from "./chart-context";
import { selectEvenlySpacedIndices } from "./x-axis";

export interface ChartNumericXAxisProps {
  valueKey?: string;
  label?: string;
  format?: (value: number) => string;
  numTicks?: number;
}

export function ChartNumericXAxis({
  valueKey = "step",
  label,
  format,
  numTicks = 6,
}: ChartNumericXAxisProps) {
  const { data, innerHeight, innerWidth, xAccessor, xScale } = useChartStable();

  const ticks = useMemo(() => {
    return selectEvenlySpacedIndices(data.length, numTicks)
      .map((index) => {
        const point = data[index];
        const raw = point?.[valueKey];
        if (!point || typeof raw !== "number" || !Number.isFinite(raw)) {
          return null;
        }
        return {
          key: `${index}-${raw}`,
          label: format ? format(raw) : String(Math.round(raw)),
          x: xScale(xAccessor(point)) ?? 0,
        };
      })
      .filter(
        (tick): tick is { key: string; label: string; x: number } =>
          tick !== null,
      );
  }, [data, format, numTicks, valueKey, xAccessor, xScale]);

  return (
    <g className="chart-numeric-x-axis" pointerEvents="none">
      {ticks.map((tick) => (
        <text
          dominantBaseline="hanging"
          fill={chartCssVars.label}
          fontSize={11}
          key={tick.key}
          textAnchor="middle"
          x={tick.x}
          y={innerHeight + 15}
        >
          {tick.label}
        </text>
      ))}
      {label ? (
        <text
          dominantBaseline="hanging"
          fill={chartCssVars.label}
          fontSize={10}
          letterSpacing={0.6}
          textAnchor="end"
          x={innerWidth}
          y={innerHeight + 31}
        >
          {label}
        </text>
      ) : null}
    </g>
  );
}

ChartNumericXAxis.displayName = "XAxis";

export interface ChartNumericYAxisProps {
  yAxisId?: string | number;
  side?: "left" | "right";
  format?: (value: number) => string;
  numTicks?: number;
}

export function ChartNumericYAxis({
  yAxisId,
  side = "left",
  format,
  numTicks = 5,
}: ChartNumericYAxisProps) {
  const { innerWidth } = useChartStable();
  const yScale = useYScale(yAxisId);
  const ticks = useMemo(
    () => yScale.ticks?.(numTicks) ?? [],
    [numTicks, yScale],
  );
  const x = side === "right" ? innerWidth + 8 : -8;

  return (
    <g className="chart-numeric-y-axis" pointerEvents="none">
      {ticks.map((tick) => (
        <text
          dominantBaseline="middle"
          fill={chartCssVars.label}
          fontSize={11}
          key={tick}
          textAnchor={side === "right" ? "start" : "end"}
          x={x}
          y={yScale(tick) ?? 0}
        >
          {format ? format(tick) : String(tick)}
        </text>
      ))}
    </g>
  );
}

ChartNumericYAxis.displayName = "YAxis";
