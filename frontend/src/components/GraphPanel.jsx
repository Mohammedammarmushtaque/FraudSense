// GraphPanel.jsx
import React, { useState, useEffect, useCallback, useRef } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  MarkerType,
  Panel,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

// ─── Constants ────────────────────────────────────────────────────────────────
const SUSPICION_COLOR = (score) => {
  if (score < 30) return { bg: "#22c55e", border: "#16a34a", text: "#fff" };
  if (score < 60) return { bg: "#f59e0b", border: "#d97706", text: "#fff" };
  return { bg: "#ef4444", border: "#dc2626", text: "#fff" };
};

const NODE_SHAPE = {
  user: "circle",
  merchant: "diamond",
  intermediate: "hexagon",
};

const CLUSTER_COLORS = [
  { bg: "rgba(99,102,241,0.08)", border: "#6366f1" },
  { bg: "rgba(20,184,166,0.08)", border: "#14b8a6" },
  { bg: "rgba(249,115,22,0.08)", border: "#f97316" },
];

// ─── Cluster Background Nodes ─────────────────────────────────────────────────
const ClusterNode = ({ data }) => (
  <div
    style={{
      width: data.width,
      height: data.height,
      background: data.bg,
      border: `2px dashed ${data.border}`,
      borderRadius: 16,
      display: "flex",
      alignItems: "flex-start",
      justifyContent: "flex-start",
      padding: "10px 14px",
      pointerEvents: "none",
    }}
  >
    <span
      style={{
        color: data.border,
        fontWeight: 700,
        fontSize: 13,
        letterSpacing: 1,
        textTransform: "uppercase",
        opacity: 0.85,
      }}
    >
      {data.label}
    </span>
  </div>
);

// ─── Custom Graph Node ────────────────────────────────────────────────────────
const GraphNode = ({ data }) => {
  const { bg, border, text } = SUSPICION_COLOR(data.suspicion_score ?? 0);
  const isPulsing = data.isPulsing;
  const type = data.node_type ?? "user";

  const shapeStyle =
    type === "merchant"
      ? {
          borderRadius: 6,
          transform: "rotate(45deg)",
          width: 38,
          height: 38,
        }
      : type === "intermediate"
      ? {
          borderRadius: "50% 0 50% 0",
          width: 38,
          height: 38,
        }
      : {
          borderRadius: "50%",
          width: 42,
          height: 42,
        };

  return (
    <div
      style={{ display: "flex", flexDirection: "column", alignItems: "center" }}
    >
      {/* Pulse ring when propagation active */}
      {isPulsing && (
        <div
          style={{
            position: "absolute",
            width: 56,
            height: 56,
            borderRadius: "50%",
            border: `2px solid ${border}`,
            animation: "pulse-ring 1.4s ease-out infinite",
            opacity: 0.6,
            top: -7,
            left: -7,
            pointerEvents: "none",
          }}
        />
      )}

      {/* Node shape */}
      <div
        style={{
          ...shapeStyle,
          background: bg,
          border: `2.5px solid ${border}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: `0 0 ${isPulsing ? 14 : 6}px ${border}55`,
          transition: "box-shadow 0.3s",
          cursor: "pointer",
        }}
      >
        <span
          style={{
            color: text,
            fontSize: 10,
            fontWeight: 700,
            transform: type === "merchant" ? "rotate(-45deg)" : "none",
            userSelect: "none",
          }}
        >
          {Math.round(data.suspicion_score ?? 0)}
        </span>
      </div>

      {/* Label */}
      <div
        style={{
          marginTop: 5,
          fontSize: 10,
          color: "#d1d5db",
          maxWidth: 70,
          textAlign: "center",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {data.label}
      </div>
    </div>
  );
};

const nodeTypes = {
  graphNode: GraphNode,
  clusterNode: ClusterNode,
};

// ─── Layout Helpers ───────────────────────────────────────────────────────────
const CLUSTER_ORIGINS = [
  { x: 60, y: 60 },
  { x: 520, y: 60 },
  { x: 290, y: 360 },
];

const layoutCluster = (members, originX, originY, cols = 4) =>
  members.map((node, i) => ({
    ...node,
    position: {
      x: originX + (i % cols) * 110 + Math.random() * 10,
      y: originY + Math.floor(i / cols) * 110 + Math.random() * 10,
    },
  }));

// ─── Main Component ───────────────────────────────────────────────────────────
export default function GraphPanel({ graphIntelligenceEnabled = false }) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [rawNodes, setRawNodes] = useState([]);
  const [rawEdges, setRawEdges] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [propagationStep, setPropagationStep] = useState(0);
  const propagationRef = useRef(null);

  // ── Fetch graph data ────────────────────────────────────────────────────────
  useEffect(() => {
    const fetchGraph = async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch("/api/graph/nodes");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setRawNodes(data.nodes ?? []);
        setRawEdges(data.edges ?? []);
      } catch (err) {
        console.error("Graph fetch failed:", err);
        setError(err.message);
        // Fall back to generated demo data
        const { nodes: demoNodes, edges: demoEdges } = generateDemoGraph();
        setRawNodes(demoNodes);
        setRawEdges(demoEdges);
      } finally {
        setLoading(false);
      }
    };
    fetchGraph();
  }, []);

  // ── Build React Flow nodes/edges from raw data ──────────────────────────────
  useEffect(() => {
    if (!rawNodes.length) return;

    // Split into 3 clusters
    const clusters = [[], [], []];
    rawNodes.forEach((n, i) => clusters[i % 3].push(n));

    // Cluster background nodes
    const clusterBgNodes = clusters.map((members, ci) => ({
      id: `cluster-${ci}`,
      type: "clusterNode",
      position: {
        x: CLUSTER_ORIGINS[ci].x - 20,
        y: CLUSTER_ORIGINS[ci].y - 20,
      },
      data: {
        label: `Cluster ${ci + 1}`,
        bg: CLUSTER_COLORS[ci].bg,
        border: CLUSTER_COLORS[ci].border,
        width: Math.ceil(members.length / 3) * 110 + 60,
        height: Math.ceil(members.length / 3) * 110 + 60,
      },
      draggable: false,
      selectable: false,
      zIndex: -1,
    }));

    // Position graph nodes per cluster
    const positionedNodes = clusters.flatMap((members, ci) =>
      layoutCluster(members, CLUSTER_ORIGINS[ci].x, CLUSTER_ORIGINS[ci].y).map(
        (n) => ({
          id: String(n.id),
          type: "graphNode",
          position: n.position,
          data: {
            ...n,
            label: n.label ?? n.id,
            suspicion_score: n.suspicion_score ?? 0,
            node_type: n.node_type ?? "user",
            isPulsing: false,
          },
          zIndex: 1,
        })
      )
    );

    // Build edges
    const flowEdges = rawEdges.map((e, i) => ({
      id: `e-${i}`,
      source: String(e.source),
      target: String(e.target),
      animated: false,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: "#6366f1",
        width: 16,
        height: 16,
      },
      style: {
        stroke: e.suspicious ? "#ef4444" : "#4b5563",
        strokeWidth: e.amount ? Math.min(1 + e.amount / 5000, 4) : 1.5,
        opacity: 0.7,
      },
      label: e.amount ? `$${e.amount.toLocaleString()}` : undefined,
      labelStyle: { fill: "#9ca3af", fontSize: 9 },
      labelBgStyle: { fill: "#1f2937", fillOpacity: 0.85 },
      data: e,
    }));

    setNodes([...clusterBgNodes, ...positionedNodes]);
    setEdges(flowEdges);
  }, [rawNodes, rawEdges]);

  // ── Propagation animation ───────────────────────────────────────────────────
  useEffect(() => {
    if (!graphIntelligenceEnabled) {
      // Clear pulsing state
      setNodes((nds) =>
        nds.map((n) =>
          n.type === "graphNode"
            ? { ...n, data: { ...n.data, isPulsing: false } }
            : n
        )
      );
      if (propagationRef.current) clearInterval(propagationRef.current);
      return;
    }

    let step = 0;
    propagationRef.current = setInterval(() => {
      step = (step + 1) % 30;
      setPropagationStep(step);
      setNodes((nds) =>
        nds.map((n) => {
          if (n.type !== "graphNode") return n;
          const idx = rawNodes.findIndex((r) => String(r.id) === n.id);
          const shouldPulse = idx === step || idx === (step + 5) % 30;
          return {
            ...n,
            data: { ...n.data, isPulsing: shouldPulse },
          };
        })
      );
    }, 600);

    return () => clearInterval(propagationRef.current);
  }, [graphIntelligenceEnabled, rawNodes]);

  // ── Minimap node color ──────────────────────────────────────────────────────
  const minimapNodeColor = useCallback((node) => {
    if (node.type === "clusterNode") return "transparent";
    const score = node.data?.suspicion_score ?? 0;
    return SUSPICION_COLOR(score).bg;
  }, []);

  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <div className="relative w-full h-full bg-gray-900 rounded-xl overflow-hidden">
      {/* Keyframe injection */}
      <style>{`
        @keyframes pulse-ring {
          0%   { transform: scale(0.9); opacity: 0.7; }
          70%  { transform: scale(1.5); opacity: 0;   }
          100% { transform: scale(1.5); opacity: 0;   }
        }
      `}</style>

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 z-50">
          <div className="flex flex-col items-center gap-3">
            <div className="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-gray-400 text-sm">Loading graph…</span>
          </div>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-50 bg-amber-900/80 border border-amber-500 text-amber-200 text-xs px-4 py-2 rounded-full flex items-center gap-2">
          <span>⚠</span>
          <span>API unavailable — showing demo data</span>
        </div>
      )}

      {/* React Flow canvas */}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.3}
        maxZoom={3}
        proOptions={{ hideAttribution: true }}
        style={{ background: "#111827" }}
      >
        <Background color="#374151" gap={20} size={1} />
        <Controls
          style={{
            background: "#1f2937",
            border: "1px solid #374151",
            borderRadius: 8,
          }}
        />
        <MiniMap
          nodeColor={minimapNodeColor}
          maskColor="rgba(0,0,0,0.6)"
          style={{
            background: "#1f2937",
            border: "1px solid #374151",
            borderRadius: 8,
          }}
        />

        {/* ── Legend Panel ── */}
        <Panel position="top-right">
          <div className="bg-gray-800/90 border border-gray-700 rounded-xl p-4 min-w-[200px] shadow-xl backdrop-blur-sm">
            <p className="text-gray-400 text-xs font-semibold uppercase tracking-widest mb-3">
              Legend
            </p>

            {/* Node types */}
            <p className="text-gray-500 text-xs mb-1.5">Node Types</p>
            <div className="flex flex-col gap-1.5 mb-3">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-indigo-500 border border-indigo-400 shrink-0" />
                <span className="text-gray-300 text-xs">User</span>
              </div>
              <div className="flex items-center gap-2">
                <div
                  className="w-4 h-4 bg-teal-500 border border-teal-400 shrink-0"
                  style={{ borderRadius: 3, transform: "rotate(45deg)" }}
                />
                <span className="text-gray-300 text-xs">Merchant</span>
              </div>
              <div className="flex items-center gap-2">
                <div
                  className="w-4 h-4 bg-orange-500 border border-orange-400 shrink-0"
                  style={{ borderRadius: "50% 0 50% 0" }}
                />
                <span className="text-gray-300 text-xs">Intermediate</span>
              </div>
            </div>

            {/* Suspicion scores */}
            <p className="text-gray-500 text-xs mb-1.5">Suspicion Score</p>
            <div className="flex flex-col gap-1.5 mb-3">
              {[
                { color: "bg-green-500", label: "Low  (<30)" },
                { color: "bg-amber-500", label: "Medium (30–60)" },
                { color: "bg-red-500", label: "High  (>60)" },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${color} shrink-0`} />
                  <span className="text-gray-300 text-xs">{label}</span>
                </div>
              ))}
            </div>

            {/* Edge types */}
            <p className="text-gray-500 text-xs mb-1.5">Edges</p>
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center gap-2">
                <div className="w-6 h-0.5 bg-gray-500" />
                <span className="text-gray-300 text-xs">Normal flow</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-0.5 bg-red-500" />
                <span className="text-gray-300 text-xs">Suspicious</span>
              </div>
            </div>

            {/* Propagation indicator */}
            {graphIntelligenceEnabled && (
              <div className="mt-3 pt-3 border-t border-gray-700 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
                <span className="text-indigo-300 text-xs font-medium">
                  Propagation active
                </span>
              </div>
            )}
          </div>
        </Panel>

        {/* ── Cluster legend ── */}
        <Panel position="bottom-left">
          <div className="flex gap-2">
            {CLUSTER_COLORS.map((c, i) => (
              <div
                key={i}
                className="flex items-center gap-1.5 bg-gray-800/80 border rounded-lg px-2.5 py-1.5 text-xs"
                style={{ borderColor: c.border, color: c.border }}
              >
                <div
                  className="w-2 h-2 rounded-sm"
                  style={{ background: c.border }}
                />
                Cluster {i + 1}
              </div>
            ))}
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
}

// ─── Demo Graph Generator (fallback) ─────────────────────────────────────────
function generateDemoGraph() {
  const nodeTypes = ["user", "merchant", "intermediate"];
  const nodes = Array.from({ length: 30 }, (_, i) => ({
    id: String(i + 1),
    label: `Node ${i + 1}`,
    node_type: nodeTypes[i % 3],
    suspicion_score: Math.round(Math.random() * 100),
  }));

  // Intra-cluster edges (dense) + inter-cluster edges (sparse)
  const edges = [];

  // Intra-cluster
  for (let c = 0; c < 3; c++) {
    const members = nodes.filter((_, i) => i % 3 === c);
    for (let i = 0; i < members.length - 1; i++) {
      edges.push({
        source: members[i].id,
        target: members[i + 1].id,
        amount: Math.round(Math.random() * 10000),
        suspicious: members[i].suspicion_score > 60,
      });
    }
  }

  // Inter-cluster bridges
  edges.push(
    { source: "1", target: "2", amount: 5000, suspicious: false },
    { source: "2", target: "3", amount: 15000, suspicious: true },
    { source: "10", target: "21", amount: 3200, suspicious: false },
    { source: "15", target: "28", amount: 8800, suspicious: true }
  );

  return { nodes, edges };
}