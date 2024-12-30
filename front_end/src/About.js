import React from 'react';
import './About.css'; // 필요에 따라 스타일 파일 추가

function About() {
    return (
        <div className="about-container">
            <h1>About Selective Morph</h1>
            <p>
                Selective Morph는 사용자가 선택한 특정 클래스에 대해 스타일 트랜스퍼를 적용하여 개인화된 이미지를 생성하는 혁신적인 플랫폼입니다.
            </p>
            <h2>비전</h2>
            <p>
                사용자의 창의성을 표현할 수 있는 공간을 제공하고, 다양한 예술적 스타일을 탐험하도록 돕습니다.
            </p>
            <h2>주요 기능</h2>
            <ul>
                <li>사용자가 직접 사진과 스타일을 업로드할 수 있습니다.</li>
                <li>선택한 클래스에만 스타일 트랜스퍼를 적용합니다.</li>
                <li>최신 AI 기술을 활용하여 정교한 결과를 제공합니다.</li>
                <li>스타일 효과의 강도를 슬라이더로 조절할 수 있습니다.</li>
                <li>실시간으로 변환 결과를 미리보기할 수 있습니다.</li>
                <li>다양한 예술 스타일 라이브러리를 제공합니다.</li>
                <li>원본 이미지의 주요 특징을 보존하며 스타일을 적용합니다.</li>
                <li>지속적인 AI 모델 업데이트로 더 나은 결과물을 제공합니다.</li>
            </ul>
            <h2>팀</h2>
            <ul>
                <li>김준래 - 프론트엔드 개발자</li>
                <li>윤현준 - ai 모델 개발자</li>
                <li>이제우 - 백엔드 개발자</li>
                <li>정세찬 - 백엔드 개발자</li>
            </ul>


            <h2>우리의 영웅들</h2>
            <p>AI가 없었다면 우리는 이 모든 것을 이룰 수 없었을 것입니다. 여기 우리의 영웅들, 즉 AI들이 있습니다:</p>
            <ul>
                <li>스타일 트랜스퍼 - 원본 이미지의 콘텐츠를 보존하면서 다른 이미지의 스타일을 전송하는 혁신적인 AI 기술입니다. 텍스처, 색상, 시각적 특징을 새로운 이미지에 적용할 수
                    있습니다.
                </li>
                <li>BiSeNet - 실시간 세그멘테이션을 위한 양방향 네트워크로, Detail Branch와 Semantic Branch를 통해 공간 정보와 의미론적 정보를 효과적으로
                    처리합니다.
                </li>
                <li>PSPNet - 이미지의 전역적 문맥 정보를 활용하는 피라미드 구조의 세그멘테이션 네트워크입니다. CityScape 데이터셋에서 80.2%의 높은 정확도를
                    달성했습니다.
                </li>
            </ul>

            <h2>참고 자료</h2>
            <ul>
                <li>
                    <a href="https://paperswithcode.com/task/style-transfer" target="_blank" rel="noopener noreferrer">CNN 스타일 트랜스퍼 관련 사이트</a>
                </li>
                <li>
                    <a href="https://arxiv.org/abs/1612.01105" target="_blank" rel="noopener noreferrer">PSP Net 논문</a>
                </li>
                <li>
                    <a href="https://arxiv.org/abs/1808.00897" target="_blank" rel="noopener noreferrer">BISE Net 논문</a>
                </li>
            </ul>

            <h2>연락처</h2>
            <p>문의사항이 있으시면 <a href="https://github.com/HuhGong/selectivemorph4">https://github.com/HuhGong</a>으로 연락주세요.
            </p>
        </div>
    );
}

export default About;
