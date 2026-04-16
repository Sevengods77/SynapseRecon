import React, { useRef, useEffect, useState } from 'react';
import { Camera } from 'lucide-react';

export const CameraFeed = ({ onFrame }) => {
  const videoRef = useRef(null);
  const [hasPermission, setHasPermission] = useState(false);

  useEffect(() => {
    let stream = null;
    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setHasPermission(true);
        }
      } catch (err) {
        console.error('Camera access denied:', err);
        setHasPermission(false);
      }
    };

    startCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="relative w-full h-full bg-industrial-800 rounded-lg overflow-hidden shadow-lg border border-industrial-700">
      {!hasPermission && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400 gap-4">
          <Camera size={48} />
          <p>Requesting camera access...</p>
        </div>
      )}
      
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={`w-full h-full object-cover transition-opacity duration-500 ${hasPermission ? 'opacity-100' : 'opacity-0'}`}
      />
      
      {/* Grid overlay for "industrial spatial mapping" aesthetic */}
      <div className="absolute inset-0 pointer-events-none opacity-20"
           style={{ backgroundImage: 'linear-gradient(#00FF9D 1px, transparent 1px), linear-gradient(90deg, #00FF9D 1px, transparent 1px)', backgroundSize: '40px 40px' }}
      ></div>
    </div>
  );
};
