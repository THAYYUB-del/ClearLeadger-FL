import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { GlobalShapFeature, IndividualShapContribution } from '../types/api';

// Calm, muted enterprise clinical palette (no neon colors)
const COLOR_TEAL = '#0F766E';
const COLOR_NAVY = '#1E3A5F';
const COLOR_SLATE = '#64748B';
const COLOR_CRIMSON = '#9F1239';
const GRID_COLOR = '#E2E8F0';

export function RoundPerformanceChart({
  data,
}: {
  data: Array<{
    round: number;
    accuracy: number;
    f1_score: number;
    roc_auc: number;
    loss: number;
  }>;
}) {
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} vertical={false} />
          <XAxis
            dataKey="round"
            tickFormatter={(v) => `R${v}`}
            tick={{ fontSize: 12, fill: '#475569' }}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
          />
          <YAxis
            domain={[60, 100]}
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 12, fill: '#475569' }}
            axisLine={false}
            tickLine={false}
            width={44}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#FFFFFF',
              borderColor: '#CBD5E1',
              borderRadius: '6px',
              fontSize: '12px',
            }}
            formatter={(value, name) => [`${Number(value ?? 0).toFixed(1)}%`, String(name ?? '')]}
            labelFormatter={(label) => `Round ${label}`}
          />
          <Legend wrapperStyle={{ fontSize: '12px' }} />
          <Line
            type="monotone"
            dataKey="accuracy"
            name="Accuracy"
            stroke={COLOR_TEAL}
            strokeWidth={2}
            dot={{ r: 3.5, fill: COLOR_TEAL }}
            activeDot={{ r: 5 }}
          />
          <Line
            type="monotone"
            dataKey="roc_auc"
            name="ROC-AUC"
            stroke={COLOR_NAVY}
            strokeWidth={2}
            dot={{ r: 3.5, fill: COLOR_NAVY }}
          />
          <Line
            type="monotone"
            dataKey="f1_score"
            name="F1 Score"
            stroke={COLOR_SLATE}
            strokeWidth={1.75}
            strokeDasharray="4 4"
            dot={{ r: 3, fill: COLOR_SLATE }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function GlobalShapChart({ features }: { features: GlobalShapFeature[] }) {
  const top12 = features.slice(0, 12);

  return (
    <div className="h-80 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={top12}
          layout="vertical"
          margin={{ top: 4, right: 20, left: 12, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} horizontal={false} />
          <XAxis
            type="number"
            tick={{ fontSize: 11, fill: '#475569' }}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="feature_name"
            width={150}
            tick={{ fontSize: 11, fill: '#334155' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#FFFFFF',
              borderColor: '#CBD5E1',
              borderRadius: '6px',
              fontSize: '12px',
            }}
            formatter={(val) => [Number(val ?? 0).toFixed(4), 'Mean |SHAP|']}
          />
          <Bar dataKey="mean_abs_shap" radius={[0, 4, 4, 0]} barSize={16}>
            {top12.map((entry) => (
              <Cell
                key={entry.feature_name}
                fill={entry.direction === 'benign_leaning' ? COLOR_TEAL : COLOR_CRIMSON}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function IndividualShapChart({
  contributions,
}: {
  contributions: IndividualShapContribution[];
}) {
  const top10 = contributions.slice(0, 10);

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={top10}
          layout="vertical"
          margin={{ top: 4, right: 20, left: 12, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} horizontal={false} />
          <XAxis
            type="number"
            tick={{ fontSize: 11, fill: '#475569' }}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="feature_name"
            width={150}
            tick={{ fontSize: 11, fill: '#334155' }}
            axisLine={false}
            tickLine={false}
          />
          <ReferenceLine x={0} stroke="#94A3B8" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#FFFFFF',
              borderColor: '#CBD5E1',
              borderRadius: '6px',
              fontSize: '12px',
            }}
            formatter={(val) => {
              const num = Number(val ?? 0);
              return [
                num >= 0 ? `+${num.toFixed(4)} (toward Benign)` : `${num.toFixed(4)} (toward Malignant)`,
                'SHAP Value',
              ];
            }}
          />
          <Bar dataKey="shap_value" radius={[3, 3, 3, 3]} barSize={16}>
            {top10.map((entry) => (
              <Cell
                key={entry.feature_name}
                fill={entry.shap_value >= 0 ? COLOR_TEAL : COLOR_CRIMSON}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ModelComparisonChart({
  rows,
}: {
  rows: Array<{
    name: string;
    accuracy: number;
    roc_auc: number;
    f1: number;
  }>;
}) {
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 11, fill: '#334155' }}
            axisLine={{ stroke: GRID_COLOR }}
            tickLine={false}
          />
          <YAxis
            domain={[50, 100]}
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 12, fill: '#475569' }}
            axisLine={false}
            tickLine={false}
            width={44}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#FFFFFF',
              borderColor: '#CBD5E1',
              borderRadius: '6px',
              fontSize: '12px',
            }}
            formatter={(value, name) => [`${Number(value ?? 0).toFixed(1)}%`, String(name ?? '')]}
          />
          <Legend wrapperStyle={{ fontSize: '12px' }} />
          <Bar dataKey="accuracy" name="Accuracy" fill={COLOR_TEAL} radius={[4, 4, 0, 0]} barSize={22} />
          <Bar dataKey="roc_auc" name="ROC-AUC" fill={COLOR_NAVY} radius={[4, 4, 0, 0]} barSize={22} />
          <Bar dataKey="f1" name="F1 Score" fill={COLOR_SLATE} radius={[4, 4, 0, 0]} barSize={22} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
