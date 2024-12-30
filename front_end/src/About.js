import React, {useState} from 'react';
import './About.css';

function About() {
    const [activeSection, setActiveSection] = useState(null);

    const features = [
        "사용자가 직접 사진과 스타일을 업로드할 수 있습니다.",
        "선택한 클래스에만 스타일 트랜스퍼를 적용합니다.",
        "최신 AI 기술을 활용하여 정교한 결과를 제공합니다.",
        "스타일 효과의 강도를 슬라이더로 조절할 수 있습니다.",
        "실시간으로 변환 결과를 미리보기할 수 있습니다.",
        "다양한 예술 스타일 라이브러리를 제공합니다.",
        "원본 이미지의 주요 특징을 보존하며 스타일을 적용합니다.",
        "지속적인 AI 모델 업데이트로 더 나은 결과물을 제공합니다."
    ];

    const teamMembers = [
        {name: "김준래", role: "프론트엔드 개발자", github: "https://github.com/junlae223"},
        {name: "윤현준", role: "AI 모델 개발자", github: "https://github.com/YHJ659"},
        {name: "이제우", role: "백엔드 개발자", github: "https://github.com/jewoos2921"},
        {name: "정세찬", role: "백엔드 개발자", github: "https://github.com/HuhGong"}
    ];

    const aiHeroes = [
        {
            name: "스타일 트랜스퍼",
            description: "원본 이미지의 콘텐츠를 보존하면서 다른 이미지의 스타일을 전송하는 혁신적인 AI 기술입니다. 텍스처, 색상, 시각적 특징을 새로운 이미지에 적용할 수 있습니다."
        },
        {
            name: "BiSeNet",
            description: "실시간 세그멘테이션을 위한 양방향 네트워크로, Detail Branch와 Semantic Branch를 통해 공간 정보와 의미론적 정보를 효과적으로 처리합니다."
        },
        {
            name: "PSPNet",
            description: "이미지의 전역적 문맥 정보를 활용하는 피라미드 구조의 세그멘테이션 네트워크입니다. CityScape 데이터셋에서 80.2%의 높은 정확도를 달성했습니다."
        }
    ];

    return (
        <div className="about-container">
            <div className="hero-section">
                <h1>About Selective Morph</h1>
                <p className="hero-description">
                    Selective Morph는 사용자가 선택한 특정 클래스에 대해 스타일 트랜스퍼를 적용하여
                    개인화된 이미지를 생성하는 혁신적인 플랫폼입니다.
                </p>
            </div>

            <section className="vision-section"
                     onMouseEnter={() => setActiveSection('vision')}
                     onMouseLeave={() => setActiveSection(null)}>
                <h2>비전</h2>
                <div className={`content-box ${activeSection === 'vision' ? 'active' : ''}`}>
                    <p>사용자의 창의성을 표현할 수 있는 공간을 제공하고, 다양한 예술적 스타일을 탐험하도록 돕습니다.</p>
                </div>
            </section>

            <section className="features-section">
                <h2>주요 기능</h2>
                <div className="features-grid">
                    {features.map((feature, index) => (
                        <div key={index} className="feature-card">
                            <span className="feature-number">{index + 1}</span>
                            <p>{feature}</p>
                        </div>
                    ))}
                </div>
            </section>

            <section className="team-section">
                <h2>팀</h2>
                <div className="team-grid">
                    {teamMembers.map((member, index) => (
                        <div key={index} className="team-card">
                            <h3>{member.name}</h3>
                            <p>{member.role}</p>
                            <a href={member.github} target="_blank" rel="noopener noreferrer">
                                GitHub Profile
                            </a>
                        </div>
                    ))}
                </div>
            </section>

            <section className="ai-heroes-section">
                <h2>우리의 영웅들</h2>
                <div className="ai-heroes-grid">
                    {aiHeroes.map((hero, index) => (
                        <div key={index} className="ai-hero-card">
                            <h3>{hero.name}</h3>
                            <p>{hero.description}</p>
                        </div>
                    ))}
                </div>
            </section>

            <section className="ai-paper-section">
                <h2>참고 자료</h2>
                <ul>
                    <li>
                        <a href="https://paperswithcode.com/task/style-transfer" target="_blank"
                           rel="noopener noreferrer">CNN
                            스타일 트랜스퍼 관련 사이트</a>
                    </li>
                    <li>
                        <a href="https://en.wikipedia.org/wiki/Neural_style_transfer" target="_blank"
                           rel="noopener noreferrer">Neural style transfer Wiki</a>
                    </li>
                    <li>
                        <a href="https://paperswithcode.com/method/pspnet" target="_blank" rel="noopener noreferrer">PSP
                            Net
                            관련 사이트</a>
                    </li>
                    <li>
                        <a href="https://arxiv.org/abs/1612.01105" target="_blank" rel="noopener noreferrer">PSP Net
                            논문</a>
                    </li>
                    <li>
                        <a href="https://arxiv.org/abs/1808.00897" target="_blank" rel="noopener noreferrer">BISE Net
                            논문</a>
                    </li>
                    <li>
                        <a href="https://github.com/CoinCheung/BiSeNet" target="_blank" rel="noopener noreferrer">BISE
                            Net
                            GitHub</a>
                    </li>
                </ul>
            </section>

            <footer className="about-footer">
                <h2>연락처</h2>
                <p>문의사항이 있으시면 <a href="https://github.com/HuhGong/selectivemorph4">GitHub</a>으로 연락주세요.</p>
            </footer>
        </div>
    );
}

export default About;
