// DOMContentLoaded 이벤트를 사용하여 DOM이 완전히 로드된 후 코드를 실행합니다.
document.addEventListener('DOMContentLoaded', () => {
    // 기본 레이아웃 설정
    const body = document.body;

    // 제목 추가
    const title = document.createElement('h1');
    title.textContent = "이미지 업로드 및 세그멘테이션";
    body.appendChild(title);

    // 파일 입력 및 버튼 추가
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.id = 'fileInput';
    fileInput.accept = 'image/*';
    body.appendChild(fileInput);

    const uploadButton = document.createElement('button');
    uploadButton.id = 'uploadButton';
    uploadButton.textContent = '업로드';
    body.appendChild(uploadButton);

    // 클래스 이미지 컨테이너 추가
    const classImagesContainer = document.createElement('div');
    classImagesContainer.id = 'classImagesContainer';
    body.appendChild(classImagesContainer);

    uploadButton.addEventListener('click', () => {
        const formData = new FormData();
        formData.append('contentImage', fileInput.files[0]);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                classImagesContainer.innerHTML = ''; // 이전 결과 초기화

                // 데이터 구조 확인
                console.log(data); // 추가: 데이터 확인

                if (data.classImages) {
                    data.classImages.forEach(imagePath => {
                        const imgElement = document.createElement('img');
                        imgElement.src = imagePath; // 이미지 경로 확인 필요
                        imgElement.style.width = '150px'; // 이미지 크기 조정
                        imgElement.style.cursor = 'pointer';
                        imgElement.addEventListener('click', () => {
                            alert(`선택한 클래스 이미지: ${imagePath}`);
                        });

                        classImagesContainer.appendChild(imgElement);
                    });
                } else {
                    console.error('No classImages found in response.');
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });

});
