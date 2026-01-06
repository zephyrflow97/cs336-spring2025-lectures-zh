import { useState, useEffect } from 'react';
import './Home.css';

function Home() {
  const [lectures, setLectures] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // 尝试获取所有可用的课件列表
    const fetchLectures = async () => {
      try {
        // 定义所有可能的课件文件（扫描 1-20 的范围）
        const possibleLectures = [];
        for (let i = 1; i <= 20; i++) {
          possibleLectures.push({
            id: i,
            name: `Lecture ${String(i).padStart(2, '0')}`,
            file: `lecture_${String(i).padStart(2, '0')}.json`
          });
        }

        // 检查每个文件是否存在
        const availableLectures = [];
        for (const lecture of possibleLectures) {
          try {
            const response = await fetch(`var/traces/${lecture.file}`);
            if (response.ok) {
              // 尝试解析JSON，确保文件内容有效
              const data = await response.json();
              if (data && data.steps) {
                availableLectures.push(lecture);
              }
            }
          } catch (e) {
            // 文件不存在或无效，跳过
          }
        }

        setLectures(availableLectures);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchLectures();
  }, []);

  if (loading) {
    return (
      <div className="home-container">
        <div className="loading">加载中...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="home-container">
        <div className="error">错误: {error}</div>
      </div>
    );
  }

  return (
    <div className="home-container">
      <header className="home-header">
        <h1>Spring 2025 课程讲座</h1>
        <p className="subtitle">选择一个讲座查看内容</p>
      </header>

      <div className="lectures-grid">
        {lectures.map((lecture) => (
          <a
            key={lecture.id}
            href={`/?trace=var/traces/${lecture.file}`}
            className="lecture-card"
          >
            <div className="lecture-number">#{lecture.id}</div>
            <div className="lecture-name">{lecture.name}</div>
            <div className="lecture-arrow">→</div>
          </a>
        ))}
      </div>

      {lectures.length === 0 && (
        <div className="no-lectures">
          <p>暂无可用的课件</p>
        </div>
      )}

      <footer className="home-footer">
        <p>共 {lectures.length} 个讲座</p>
      </footer>
    </div>
  );
}

export default Home;

