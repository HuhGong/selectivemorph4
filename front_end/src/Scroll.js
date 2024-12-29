// ImageCompare.jsx
import React, {useRef, useState, useEffect} from "react";
import "./Scroll.css";
import image0 from './assets/samples/sample_0.jpg';
import image1 from './assets/samples/sample_1.png';
import image2 from './assets/samples/sample_3.png';
import image3 from './assets/samples/sample_4.png';
import image4 from './assets/samples/sample_6.png';

const ImageCompare = () => {
    const containerRef = useRef(null);
    const [isDragging, setIsDragging] = useState(false);
    const [percentage, setPercentage] = useState(50);
    const [currentImage, setCurrentImage] = useState(0);

    const imageUrls = [image0, image1, image2, image3, image4];

    const calculatePercentage = (clientX) => {
        const {left, width} = containerRef.current.getBoundingClientRect();
        const position = clientX - left;
        const percentage = (position / width) * 100;
        return Math.min(Math.max(percentage, 0), 100);
    };

    const handleMouseDown = () => {
        setIsDragging(true);
    };

    const handleMouseMove = (e) => {
        if (!isDragging) return;
        setPercentage(calculatePercentage(e.clientX));
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    const handleTouchMove = (e) => {
        if (!isDragging) return;
        setPercentage(calculatePercentage(e.touches[0].clientX));
    };

    useEffect(() => {
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
        document.addEventListener('touchmove', handleTouchMove);
        document.addEventListener('touchend', handleMouseUp);

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            document.removeEventListener('touchmove', handleTouchMove);
            document.removeEventListener('touchend', handleMouseUp);
        };
    }, [isDragging]);

    return (
        <div className="style-transfer-container">
            <div className="hero-section">
                <h1>AI Style Transfer</h1>
                <p>Experience the magic of neural style transfer</p>
            </div>

            <div className="image-compare" ref={containerRef}>
                <div className="image-wrapper">
                    <img
                        className="image-left"
                        src={image0}
                        alt="Original"
                    />
                    <img
                        className="image-right"
                        src={imageUrls[currentImage]}
                        alt="Styled"
                        style={{
                            clipPath: `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`
                        }}
                    />
                </div>
                <div
                    className="slider"
                    style={{left: `${percentage}%`}}
                    onMouseDown={handleMouseDown}
                    onTouchStart={handleMouseDown}
                >
                    <div className="slider-button"/>
                </div>
            </div>

            <div className="button-container">
                {imageUrls.map((_, index) => (
                    <button
                        key={index}
                        className={`style-button ${currentImage === index ? 'active' : ''}`}
                        onClick={() => setCurrentImage(index)}
                    >
                        Style {index + 1}
                    </button>
                ))}
            </div>

            <div className="gallery-section">
                <h2>Style Gallery</h2>
                <div className="gallery-grid">
                    {imageUrls.map((url, index) => (
                        <div
                            key={index}
                            className="gallery-item"
                            onClick={() => setCurrentImage(index)}
                        >
                            <img src={url} alt={`Style ${index + 1}`}/>
                            <div className="gallery-overlay">
                                <span>Style {index + 1}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="features-section">
                <h2>Features</h2>
                <div className="features-grid">
                    <div className="feature-card">
                        <div className="feature-icon">🎨</div>
                        <h3>Multiple Styles</h3>
                        <p>Choose from various artistic styles to transform your images</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">⚡</div>
                        <h3>Real-time Preview</h3>
                        <p>See the transformation happen instantly as you slide</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">📱</div>
                        <h3>Responsive Design</h3>
                        <p>Works perfectly on all devices and screen sizes</p>
                    </div>
                </div>
            </div>

            <div className="how-it-works">
                <h2>How It Works</h2>
                <div className="steps-container">
                    <div className="step">
                        <div className="step-number">1</div>
                        <h3>Upload Image</h3>
                        <p>Select your favorite photo to transform</p>
                    </div>
                    <div className="step">
                        <div className="step-number">2</div>
                        <h3>Choose Style</h3>
                        <p>Pick from our curated collection of artistic styles</p>
                    </div>
                    <div className="step">
                        <div className="step-number">3</div>
                        <h3>Compare Segmentation</h3>
                        <p>Use the slider to compare Segmentation for style transfer</p>
                    </div>
                    <div className="step">
                        <div className="step-number">4</div>
                        <h3>Compare Results</h3>
                        <p>Use the slider to compare original and styled versions</p>
                    </div>
                </div>
            </div>


        </div>

    );
};

export default ImageCompare;
