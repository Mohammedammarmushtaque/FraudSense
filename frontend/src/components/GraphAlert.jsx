import React, { useMemo, useState, useEffect, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { motion } from "framer-motion";

export default function GraphAlert({ coordinatedAttack, graphStats }) {
  const hasAttack = coordinatedAttack && coordinatedAttack.coordinated;
  const containerRef = useRef(null);
  const [dim, setDim] = useState({ w: 300, h: 250 });

  useEffect(() => {
    if (containerRef.current) {
      setDim({
        w: containerRef.current.offsetWidth,
        h: containerRef.current.offsetHeight || 250 
      });
    }
    const handleResize = () => {
      if (containerRef.current) {
        setDim({
          w: containerRef.current.offsetWidth,
          h: containerRef.current.offsetHeight || 250 
        });
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [hasAttack]);

  // Simulate graph data dynamically based on context
  const graphData = useMemo(() => {
    const nodes = [];
    const links = [];
    
    // Core Merchant Node
    const targetMerch = coordinatedAttack?.merchant_id || "MERCHANT-HUB";
    nodes.push({ 
      id: targetMerch, 
      group: "merchant", 
      name: "Target Merchant", 
      color: hasAttack ? "#ef4444" : "#10b981", 
      val: hasAttack ? 8 : 4
    });

    if (hasAttack) {
      // 🚨 DENSE ATTACK CLUSTER (Sybil/Coordinated)
      const size = Math.max(5, coordinatedAttack.attack_size || 8);
      
      // Shared Compromised Device
      nodes.push({ id: "shared_dev", name: "Compromised Device", color: "#991b1b", val: 5 });
      
      for (let i = 0; i < size; i++) {
        const uNode = `bot_${i}`;
        nodes.push({ id: uNode, name: `Attacker User ${i}`, color: "#f87171", val: 2.5 });
        
        // Link compromised device to all fake user accounts
        links.push({ source: "shared_dev", target: uNode, color: "#991b1b" });
        // Link all fake user accounts to target merchant concurrently
        links.push({ source: uNode, target: targetMerch, color: "#ef4444" });
      }
    } else {
      // ✅ NORMAL SPARSE NETWORK
      for (let i = 0; i < 6; i++) {
        const usr = `usr_${i}`;
        const dev = `dev_${i}`;
        nodes.push({ id: usr, name: `Normal User ${i}`, color: "#34d399", val: 1.5 });
        nodes.push({ id: dev, name: `Trusted Device ${i}`, color: "#059669", val: 1 });
        
        // Clean 1-1 mappings
        links.push({ source: dev, target: usr, color: "#10b981" });
        
        // Sparse connection to the hub
        if (Math.random() > 0.4) {
          links.push({ source: usr, target: targetMerch, color: "#10b981" });
        }
      }
    }

    return { nodes, links };
  }, [hasAttack, coordinatedAttack]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <span className="text-base">🌐</span>
        <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest">Graph Intelligence</h3>
        {hasAttack && (
          <span className="ml-auto text-xs font-bold text-red-300 bg-red-900/30 border border-red-700/50 px-2 py-0.5 rounded-full animate-pulse">
            ⚡ ATTACK DETECTED
          </span>
        )}
      </div>

      <div 
        className={`relative rounded-xl border p-1 overflow-hidden transition-colors duration-500 flex flex-col items-center ${
          hasAttack ? "border-red-800/60 bg-red-950/20" : "border-gray-800/60 bg-gray-900/40"
        }`}
      >
        {/* Analytics overlay */}
        <div className="absolute top-3 left-3 z-10 pointer-events-none flex flex-col gap-1">
          {hasAttack ? (
            <>
              <span className="text-[10px] font-black text-red-500 uppercase tracking-wider bg-red-950/80 px-2 py-0.5 rounded">SYBIL DETECTED</span>
              <span className="text-[9px] text-red-300 bg-red-900/40 px-2 py-0.5 rounded">Clustered Subgraphs: 1</span>
              <span className="text-[9px] text-red-300 bg-red-900/40 px-2 py-0.5 rounded">Anomalous Density: 94%</span>
            </>
          ) : (
            <>
              <span className="text-[10px] font-black text-emerald-500 uppercase tracking-wider bg-emerald-950/80 px-2 py-0.5 rounded">NETWORK SECURE</span>
              <span className="text-[9px] text-emerald-300 bg-emerald-900/40 px-2 py-0.5 rounded">Graph Density: Normal</span>
              <span className="text-[9px] text-emerald-300 bg-emerald-900/40 px-2 py-0.5 rounded">Orphans: 0</span>
            </>
          )}
        </div>

        {/* Graph Container */}
        <div ref={containerRef} className="w-full h-[240px] flex items-center justify-center opacity-80 mix-blend-screen">
          <ForceGraph2D
            width={dim.w - 10} // Padding offset
            height={dim.h}
            graphData={graphData}
            nodeColor="color"
            nodeVal="val"
            linkColor="color"
            linkWidth={hasAttack ? 1.5 : 1}
            linkDirectionalParticles={hasAttack ? 4 : 2}
            linkDirectionalParticleWidth={hasAttack ? 2.5 : 1.5}
            linkDirectionalParticleSpeed={hasAttack ? 0.012 : 0.004}
            linkDirectionalParticleColor={() => hasAttack ? "#fca5a5" : "#6ee7b7"}
            backgroundColor="#00000000"
            enableZoomInteraction={true}    // Slightly interactive
            enablePanInteraction={true}
            d3AlphaDecay={0.05}             // Keep it moving smoothly
            d3VelocityDecay={0.2}
          />
        </div>
      </div>
    </div>
  );
}
