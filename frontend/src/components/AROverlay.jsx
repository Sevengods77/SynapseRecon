import React, { useMemo } from 'react';

export const AROverlay = ({ fragments, referenceImage }) => {
  // Render Ghost Layer and smart snap bounding boxes
  const renderFragments = useMemo(() => {
    return fragments.map((frag) => {
      const isCorrect = frag.isSnapped || (frag.distance !== undefined && frag.distance < 10 && Math.abs(frag.r) < 10);
      
      return (
        <div 
          key={frag.fragmentId}
          className={`absolute border-2 transition-all duration-300 ${isCorrect ? 'border-green-500 shadow-[0_0_15px_rgba(0,255,157,0.8)]' : 'border-industrial-warning'}`}
          style={{
            left: frag.tx ? 300 + frag.tx : 300, // mock base coordinates
            top: frag.ty ? 200 + frag.ty : 200,
            width: '100px',
            height: '100px',
            transform: `translate(-50%, -50%) rotate(${frag.r || 0}deg)`,
          }}
        >
          {isCorrect && (
            <div className="absolute -top-6 left-1/2 -translate-x-1/2 bg-industrial-900 text-green-500 text-xs px-2 py-1 rounded shadow">
              Snapped
            </div>
          )}
        </div>
      );
    });
  }, [fragments]);

  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Ghost Layer: semi-transparent reference image */}
      {referenceImage && (
        <img 
          src={referenceImage} 
          alt="Ghost Layer" 
          className="absolute inset-0 w-full h-full object-cover opacity-30 select-none"
        />
      )}

      {/* Fragment projections */}
      <div className="relative w-full h-full">
        {renderFragments}
      </div>
    </div>
  );
};
