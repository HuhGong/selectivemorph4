import React, {useState} from "react";
import {FaUpload} from "react-icons/fa6";
import {FaImage} from "react-icons/fa";

function PhotoSelector() {
    const [contentImage, setContentImage] = useState(null);
    const [styleImage, setStyleImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [outputImages, setOutputImages] = useState([]); // 여러 출력 이미지 저장
    const [currentSelection, setCurrentSelection] = useState(''); // 현재 선택된 이미지 상태 ('content' 또는 'style')

    const [contentPhotos, setContentPhotos] = useState([
        {id: 1, src: '', isUpload: true},
        {id: 2, src: require('./assets/content/image4.jpg')},
        {id: 3, src: require('./assets/content/image5.jpg')},
        {id: 4, src: require('./assets/content/image6.jpeg')},
        {id: 5, src: require('./assets/content/image8.png')},
        {id: 6, src: require('./assets/content/image11.jpg')},
    ]);

    const [stylePhotos, setStylePhotos] = useState([
        {id: 1, src: '', isUpload: true},
        {id: 2, src: require('./assets/style/image1.jpg')},
        {id: 3, src: require('./assets/style/image2.jpg')},
        {id: 4, src: require('./assets/style/image3.jpg')},
        {id: 5, src: require('./assets/style/image4.jpg')},
        {id: 6, src: require('./assets/style/image6.jpg')},
    ]);

    const handleImageUpload = (e, type) => {
        if (e.target.files && e.target.files[0]) {
            const reader = new FileReader();
            reader.onload = (event) => {
                if (type === 'content') {
                    setContentImage(event.target.result);
                } else if (type === 'style') {
                    setStyleImage(event.target.result);
                }
            };
            reader.readAsDataURL(e.target.files[0]);
        }
    };

    const handlePhotoClick = (photo, type) => {
        if (photo.isUpload) {
            document.getElementById(`${type}UploadInput`).click();
        } else {
            if (type === 'content') {
                setContentImage(photo.src);
            } else if (type === 'style') {
                setStyleImage(photo.src);
            }
        }
    };

    const handleFileUploadToBackend = async () => {
        if (!contentImage || !styleImage) {
            alert('Content Image와 Style Image를 모두 선택해 주세요.');
            return;
        }

        setLoading(true);
        // 업로드 전에 outputImages 배열을 비움
        setOutputImages([]);

        try {
            const response = await fetch('http://localhost:5000/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    contentImage: contentImage.startsWith('data:image/') ? contentImage : await convertToBase64(contentImage),
                    styleImage: styleImage.startsWith('data:image/') ? styleImage : await convertToBase64(styleImage),
                }),
            });

            if (response.ok) {
                const data = await response.json();
                alert('이미지 업로드가 완료되었습니다!');

                if (data.outputImages) {
                    const absoluteImagePaths = data.outputImages.map(imagePath =>
                        `http://localhost:5000${imagePath}`
                    );
                    // 새로운 이미지로 상태를 완전히 교체
                    setOutputImages(absoluteImagePaths);
                } else {
                    console.error('Output images are undefined.');
                    alert('서버에서 출력 이미지를 받지 못했습니다.');
                }
            } else {
                const errorData = await response.json();
                alert(`이미지 업로드 실패: ${errorData.message || response.statusText}`);
                console.error('Upload error:', errorData);
            }
        } catch (error) {
            alert('이미지 업로드 중 오류가 발생했습니다.');
            console.error('Fetch error:', error);
        } finally {
            setLoading(false);
        }
    };


    // 이미지 경로를 Base64로 변환하는 함수
    const convertToBase64 = (url) => {
        return new Promise((resolve, reject) => {
            fetch(url)
                .then((res) => res.blob())
                .then((blob) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result);
                    reader.onerror = (err) => reject(err);
                    reader.readAsDataURL(blob);
                })
                .catch((err) => reject(err));
        });
    };

    const toggleSelection = (type) => {
        setCurrentSelection(type); // 'content' 또는 'style'을 현재 선택으로 설정
    };

    return (
        <div className="container">
            {/* Left Section */}
            <div className="left-section">
                <div className="image-box">
                    <h3>Content Image</h3>
                    <div
                        className={`image-preview-container ${
                            currentSelection === 'content' ? 'selected' : ''
                        }`}
                        onClick={() => toggleSelection('content')}
                    >
                        {contentImage ? (
                            <img src={contentImage} alt="Content" className="image-preview"/>
                        ) : (
                            <div className="placeholder">
                                <FaImage style={{fontSize: '24px', color: '#888'}}/>
                            </div>
                        )}
                    </div>
                    <input
                        type="file"
                        accept="image/*"
                        style={{display: 'none'}}
                        id="contentUploadInput"
                        onChange={(e) => handleImageUpload(e, 'content')}
                    />
                </div>

                <div className="image-box">
                    <h3>Style Image</h3>
                    <div
                        className={`image-preview-container ${
                            currentSelection === 'style' ? 'selected' : ''
                        }`}
                        onClick={() => toggleSelection('style')}
                    >
                        {styleImage ? (
                            <img src={styleImage} alt="Style" className="image-preview"/>
                        ) : (
                            <div className="placeholder">
                                <FaImage style={{fontSize: '24px', color: '#888'}}/>
                            </div>
                        )}
                    </div>
                    <input
                        type="file"
                        accept="image/*"
                        style={{display: 'none'}}
                        id="styleUploadInput"
                        onChange={(e) => handleImageUpload(e, 'style')}
                    />
                    {/* Upload Button */}
                    <button type="button" className="btn btn-primary" onClick={handleFileUploadToBackend}
                            disabled={loading}>
                        {loading ? '업로드 중...' : '스타일 전송을 위한 업로드'}
                    </button>
                    <div className="right-section">
                        {outputImages.length > 0 && (
                            <div className="result-container">
                                <h4>생성된 이미지:</h4>
                                {outputImages.map((image, index) => (
                                    <div key={index}>
                                        <img
                                            src={image}
                                            alt={`Generated Output ${index}`}
                                            className="result-preview1"
                                        />
                                        <button className="download-button">Download</button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Right Section */}
            <div className="right-section">
                {currentSelection === 'content' && (
                    <>
                        <h3>Content Gallery</h3>
                        <div className="gallery">
                            {contentPhotos.map((photo) => (
                                <div
                                    key={photo.id}
                                    onClick={() => handlePhotoClick(photo, 'content')}
                                    className="thumbnail"
                                >
                                    {photo.isUpload ? (
                                        <div className="thumbnail">
                                            <FaUpload style={{fontSize: '24px', color: '#888'}}/>
                                        </div>
                                    ) : (
                                        <img src={photo.src} alt="Thumbnail" className="thumbnail-image"/>
                                    )}
                                </div>
                            ))}
                            <input
                                id="contentUploadInputGallery"
                                type="file"
                                accept="image/*"
                                style={{display: 'none'}}
                                onChange={(e) => handleImageUpload(e, 'content')}
                            />
                        </div>
                    </>
                )}

                {currentSelection === 'style' && (
                    <>
                        <h3>Style Gallery</h3>
                        <div className="gallery">
                            {stylePhotos.map((photo) => (
                                <div
                                    key={photo.id}
                                    onClick={() => handlePhotoClick(photo, 'style')}
                                    className="thumbnail"
                                >
                                    {photo.isUpload ? (
                                        <div className="thumbnail">
                                            <FaUpload style={{fontSize: '24px', color: '#888'}}/>
                                        </div>
                                    ) : (
                                        <img src={photo.src} alt="Thumbnail" className="thumbnail-image"/>
                                    )}
                                </div>
                            ))}
                            <input
                                id="styleUploadInputGallery"
                                type="file"
                                accept="image/*"
                                style={{display: 'none'}}
                                onChange={(e) => handleImageUpload(e, 'style')}
                            />
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}

export default PhotoSelector;
