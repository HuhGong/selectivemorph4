import React, { useRef, useState, useEffect } from "react";
import "./scroll.css";
import image0 from './assets/samples/sample_0.jpg'; // 이미지 가져오기
import image1 from './assets/samples/sample_1.png'; // 이미지 가져오기
import image2 from './assets/samples/sample_3.png'; // 추가 이미지 가져오기
import image3 from './assets/samples/sample_4.png'; // 추가 이미지 가져오기
import image4 from './assets/samples/sample_6.png'; // 추가 이미지 가져오기

const ImageCompare = () => {
    const containerRef = useRef(null);
    const sliderRef = useRef(null);
    const [isDragging, setIsDragging] = useState(false);
    const [percentage, setPercentage] = useState(50);
    const [currentImage, setCurrentImage] = useState(0);
    const imageUrls = [
        image0, // 로컬 이미지 사용
        image1, // 로컬 이미지 사용
        image2, // 로컬 이미지 사용
        image3, // 로컬 이미지 사용
        image4  // 로컬 이미지 사용
    ];

    const handleMouseMove = (e) => {
        if (!isDragging || !containerRef.current) return;

        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const containerWidth = containerRef.current.offsetWidth;

        const newPercentage = Math.max(0, Math.min((x / containerWidth) * 100, 100));
        setPercentage(newPercentage);
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    const handleTouchMove = (e) => {
        if (!isDragging || !containerRef.current) return;

        const touch = e.touches[0];
        const rect = containerRef.current.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const containerWidth = containerRef.current.offsetWidth;

        const newPercentage = Math.max(0, Math.min((x / containerWidth) * 100, 100));
        setPercentage(newPercentage);
    };

    const changeImage = (index) => {
        setCurrentImage(index);
    }

    useEffect(() => {
        document.addEventListener("mousemove", handleMouseMove);
        document.addEventListener("mouseup", handleMouseUp);
        document.addEventListener("touchmove", handleTouchMove);
        document.addEventListener("touchend", handleMouseUp);

        return () => {
            document.removeEventListener("mousemove", handleMouseMove);
            document.removeEventListener("mouseup", handleMouseUp);
            document.removeEventListener("touchmove", handleTouchMove);
            document.removeEventListener("touchend", handleMouseUp);
        };
    }, [isDragging]);

    return (
        <>
            <div className="image-compare" ref={containerRef}>
                <div className="image-wrapper">
                    <img
                        className="image-left"
                        src={image0} // 로컬 이미지 사용
                        alt="이미지1"
                    />
                    <img
                        className="image-right"
                        src={imageUrls[currentImage]}
                        alt="이미지2"
                        style={{
                            clipPath: `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`
                        }}
                    />
                </div>
                <div
                    className="slider"
                    ref={sliderRef}
                    style={{ left: `${percentage}%` }}
                    onMouseDown={() => setIsDragging(true)}
                    onTouchStart={() => setIsDragging(true)}
                >
                    <div className="slider-button" />
                </div>
            </div>
            <div className="button-container">
                {imageUrls.map((_, index) => (
                    <button
                        key={index}
                        className={`image-button ${currentImage === index ? 'active' : ''}`}
                        onClick={() => changeImage(index)}
                    >
                        이미지 {index + 1}
                    </button>
                ))}
            </div>
        </>
    );
};

export default ImageCompare;
