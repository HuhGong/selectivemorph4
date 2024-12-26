import React, { useEffect, useState } from 'react';

function AnnotationViewer() {
    const [annotations, setAnnotations] = useState([]);

    useEffect(() => {
        const fetchAnnotations = async () => {
            try {
                const response = await fetch('http://localhost:5000/annotations'); // 서버 주소에 맞게 수정
                const data = await response.json();
                setAnnotations(data.annotations);
            } catch (error) {
                console.error('Error fetching annotations:', error);
            }
        };

        fetchAnnotations();
    }, []);

    return (
        <div className="annotation-viewer">
            <h2>주석 이미지</h2>
            <div className="annotation-images">
                {annotations.map((annotation, index) => (
                    <img key={index} src={annotation} alt={`Annotation ${index}`} />
                ))}
            </div>
        </div>
    );
}

export default AnnotationViewer;
