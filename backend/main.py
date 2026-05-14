from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import io
import base64
from datetime import datetime
import os
import traceback
import json
from dotenv import load_dotenv

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
try:
    import dashscope
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
load_dotenv()
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

app = FastAPI(title="概率统计AI动态演示平台API", version="1.0.0")

# 将第33-39行的 CORS 配置改为:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 用户会话数据存储文件
USER_DATA_FILE = "user_sessions.json"

def load_user_data():
    """加载用户数据"""
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_data(data):
    """保存用户数据到JSON文件"""
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class UserLogin(BaseModel):
    name: str
    student_id: str

class ChapterTimeRecord(BaseModel):
    student_id: str
    chapter: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None

class LoginResponse(BaseModel):
    success: bool
    message: str
    user_info: Optional[Dict[str, str]] = None





class PoissonApproxRequest(BaseModel):
    n: int = Field(ge=10, le=1000, default=100)
    p: float = Field(ge=0.01, le=0.1, default=0.05)

class NormalApproxRequest(BaseModel):
    n: int = Field(ge=10, le=1000, default=50)
    p: float = Field(ge=0.1, le=0.9, default=0.5)

class DigitalFeaturesRequest(BaseModel):
    dist_type: str = Field(default="正态分布")
    n_samples: int = Field(ge=100, le=50000, default=10000)
    mean: float = Field(default=0.0)
    std: float = Field(default=1.0)
    min_val: float = Field(default=0.0)
    max_val: float = Field(default=1.0)
    lam: float = Field(default=5.0)
    rate: float = Field(default=1.0)
    binom_n: int = Field(default=10)
    binom_p: float = Field(default=0.5)

class OrderStatsRequest(BaseModel):
    dist_type: str = Field(default="正态分布")
    sample_size: int = Field(ge=5, le=50, default=15)

class MomentEstRequest(BaseModel):
    a_true: float = Field(default=0.0)
    b_true: float = Field(default=1.0)
    n: int = Field(ge=2, le=500, default=100)

class MLEEstRequest(BaseModel):
    mu_true: float = Field(default=5.0)
    sigma_true: float = Field(ge=0.1, le=5.0, default=2.0)
    n: int = Field(ge=1, le=500, default=100)

class CIAnimationRequest(BaseModel):
    sample_size: int = Field(ge=2, le=1000, default=50)
    confidence_level: float = Field(ge=0.80, le=0.99, default=0.95)

class CoinTossRequest(BaseModel):
    n_coin: int = Field(ge=1, le=1000, default=100)


class NeedleRequest(BaseModel):
    n_needle: int = Field(ge=10, le=5000, default=1000)
    L: float = Field(ge=0.1, le=2.0, default=1.0)
    D: float = Field(ge=0.5, le=3.0, default=2.0)


class EventRelationRequest(BaseModel):
    probA: float = Field(ge=0.0, le=1.0, default=0.5)
    probB: float = Field(ge=0.0, le=1.0, default=0.5)
    probAB: float = Field(ge=0.0, le=1.0, default=0.3)


class DiceRollRequest(BaseModel):
    n_dice: int = Field(ge=10, le=1000, default=100)


class GeometricProbRequest(BaseModel):
    a: float = Field(ge=0.1, le=1.0, default=0.5)
    b: float = Field(ge=0.1, le=1.0, default=0.5)


class ConditionalProbRequest(BaseModel):
    pA: float = Field(ge=0.01, le=0.99, default=0.5)
    pB: float = Field(ge=0.01, le=0.99, default=0.5)
    pAB: float = Field(ge=0.01, le=1.0, default=0.25)


class DistributionRequest(BaseModel):
    dist_type: str
    params: Dict[str, float]
    n_samples: Optional[int] = 10000


class CLTRequest(BaseModel):
    dist_type: str = "均匀分布"
    n_samples: int = Field(ge=10, le=1000, default=30)
    n_trials: int = Field(ge=100, le=10000, default=1000)


class LLNRequest(BaseModel):
    dist_type: str = "均匀分布"
    num_trials: int = Field(ge=10, le=1000, default=100)


class SamplingDistRequest(BaseModel):
    mu: float = 0.0
    sigma: float = 1.0
    n: int = Field(ge=5, le=100, default=30)


class OrderStatsRequest(BaseModel):
    dist_type: str = "正态分布"
    sample_size: int = Field(ge=5, le=100, default=20)


class TTestRequest(BaseModel):
    sample_size: int = Field(ge=10, le=1000, default=50)
    hypothesized_mean: float = 0.0
    true_mean: float = 0.0
    alpha: float = Field(ge=0.01, le=0.1, default=0.05)


class TwoSampleTTestRequest(BaseModel):
    sample_size1: int = Field(ge=10, le=1000, default=50)
    sample_size2: int = Field(ge=10, le=1000, default=50)
    mean1: float = 0.0
    mean2: float = 2.0
    std1: float = 1.0
    std2: float = 1.0
    alpha: float = Field(ge=0.01, le=0.1, default=0.05)
    equal_var: bool = True


class ConfidenceIntervalRequest(BaseModel):
    sample_size: int = Field(ge=10, le=1000, default=50)
    confidence_level: float = Field(ge=0.8, le=0.99, default=0.95)


class MomentEstimationRequest(BaseModel):
    n: int = Field(ge=10, le=1000, default=100)


class MLEstimationRequest(BaseModel):
    n: int = Field(ge=10, le=1000, default=100)


class EstimatorEfficiencyRequest(BaseModel):
    dist_type: str = "正态分布"
    sample_size: int = Field(ge=10, le=1000, default=50)


class JointDiscreteRequest(BaseModel):
    prob00: float = 0.25
    prob01: float = 0.25
    prob10: float = 0.25
    prob11: float = 0.25


class JointContinuousRequest(BaseModel):
    dist_type_x: str = "正态分布"
    dist_type_y: str = "正态分布"
    corr_coef: float = 0.0


class AIMessage(BaseModel):
    message: str
    context: Optional[str] = None

class CodeExecutionRequest(BaseModel):
    code: str

def create_animation_frames(figures, duration=125):
    """将多帧matplotlib figure转换为base64 PNG帧数组"""
    frames = []
    for fig in figures:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        frames.append(f"data:image/png;base64,{img_base64}")
        buf.close()
        plt.close(fig)
    return frames


def create_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

def create_animation_frames_v2(figures):
    """将多帧matplotlib figure转换为base64 PNG帧数组（优化版）"""
    frames = []
    for fig in figures:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        frames.append(f"data:image/png;base64,{img_base64}")
        buf.close()
        plt.close(fig)
    return frames



@app.get("/")
async def root():
    return {"message": "概率学习平台API", "version": "1.0.0"}


@app.post("/api/user/login")
async def user_login(request: UserLogin):
    """用户登录"""
    try:
        if not request.name or not request.student_id:
            raise HTTPException(status_code=400, detail="姓名和学号不能为空")

        user_data = load_user_data()

        # 初始化用户数据（如果不存在）
        if request.student_id not in user_data:
            user_data[request.student_id] = {
                "name": request.name,
                "student_id": request.student_id,
                "login_time": datetime.now().isoformat(),
                "chapters": {}
            }
            save_user_data(user_data)

        return {
            "success": True,
            "message": "登录成功",
            "user_info": {
                "name": request.name,
                "student_id": request.student_id
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"登录失败: {str(e)}")


@app.post("/api/user/chapter/start")
async def chapter_start(request: ChapterTimeRecord):
    """记录章节学习开始时间"""
    try:
        user_data = load_user_data()

        if request.student_id not in user_data:
            raise HTTPException(status_code=404, detail="用户未登录")

        # 记录开始时间
        user_data[request.student_id]["chapters"][request.chapter] = {
            "start_time": request.start_time,
            "end_time": None,
            "duration_seconds": None,
            "status": "learning"
        }

        save_user_data(user_data)

        return {
            "success": True,
            "message": "开始时间已记录"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记录失败: {str(e)}")


@app.post("/api/user/chapter/end")
async def chapter_end(request: ChapterTimeRecord):
    """记录章节学习结束时间"""
    try:
        user_data = load_user_data()

        if request.student_id not in user_data:
            raise HTTPException(status_code=404, detail="用户未登录")

        if request.chapter not in user_data[request.student_id]["chapters"]:
            raise HTTPException(status_code=404, detail="未找到章节开始记录")

        # 计算持续时间
        start_time = datetime.fromisoformat(request.start_time)
        end_time = datetime.fromisoformat(request.end_time)
        duration = (end_time - start_time).total_seconds()

        # 更新结束时间和持续时间
        user_data[request.student_id]["chapters"][request.chapter].update({
            "end_time": request.end_time,
            "duration_seconds": duration,
            "status": "completed"
        })

        save_user_data(user_data)

        return {
            "success": True,
            "message": "结束时间已记录",
            "duration_seconds": duration
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记录失败: {str(e)}")


@app.get("/api/user/export/{student_id}")
async def export_user_data(student_id: str):
    """导出用户学习数据为Excel"""
    try:
        user_data = load_user_data()

        if student_id not in user_data:
            raise HTTPException(status_code=404, detail="用户不存在")

        user = user_data[student_id]

        # 准备数据
        rows = []
        for chapter, info in user["chapters"].items():
            start_time = info.get("start_time", "")
            end_time = info.get("end_time", "")
            duration = info.get("duration_seconds", 0)

            # 格式化时间
            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                start_str = ""

            if end_time:
                end_dt = datetime.fromisoformat(end_time)
                end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                end_str = ""

            # 格式化持续时间
            if duration:
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                duration_str = ""

            rows.append({
                "姓名": user["name"],
                "学号": user["student_id"],
                "章节": chapter,
                "开始时间": start_str,
                "结束时间": end_str,
                "学习时长": duration_str,
                "状态": "已完成" if info.get("status") == "completed" else "学习中"
            })

        # 创建DataFrame
        df = pd.DataFrame(rows)

        # 生成Excel文件
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='学习记录')

        output.seek(0)

        # 转换为base64
        excel_base64 = base64.b64encode(output.getvalue()).decode('utf-8')

        return {
            "success": True,
            "filename": f"{user['name']}_{student_id}_学习记录.xlsx",
            "data": excel_base64
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")


@app.get("/api/user/sessions/all")
async def get_all_sessions():
    """获取所有用户的学习数据（用于管理员查看）"""
    try:
        user_data = load_user_data()
        return {
            "success": True,
            "users": user_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据失败: {str(e)}")


@app.post("/api/chapter1/coin-toss")
async def coin_toss(request: CoinTossRequest):
    results = ["正面" if np.random.random() > 0.5 else "反面" for _ in range(request.n_coin)]
    heads = results.count("正面")
    tails = results.count("反面")

    fig, ax = plt.subplots()
    ax.bar(["正面", "反面"], [heads, tails], color=['red', 'blue'])
    ax.set_title("投币结果分布")
    ax.set_ylabel("出现次数")
    img = create_plot(fig)

    return {
        "heads": heads,
        "tails": tails,
        "heads_freq": heads / request.n_coin,
        "tails_freq": tails / request.n_coin,
        "image": img
    }


@app.post("/api/chapter1/coin-toss-animation")
async def coin_toss_animation(request: CoinTossRequest):
    """投币实验动画 - 返回帧序列"""
    try:
        n_frames = min(request.n_coin // 10, 50)
        frame_interval = max(request.n_coin // n_frames, 1)

        heads_count = 0
        tails_count = 0
        figures = []

        for i in range(1, request.n_coin + 1):
            if np.random.random() > 0.5:
                heads_count += 1
            else:
                tails_count += 1

            if i % frame_interval == 0 or i == request.n_coin:
                fig, ax = plt.subplots(figsize=(8, 5))
                categories = ['正面', '反面']
                values = [heads_count, tails_count]
                colors = ['#FF6B6B', '#4ECDC4']

                bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                ax.set_ylim(0, max(values + [1]) * 1.3)
                ax.set_title(f'投币实验 (第 {i} 次)\n正面频率: {heads_count / i:.3f}', fontsize=13, fontweight='bold')
                ax.set_ylabel('出现次数', fontsize=11)
                ax.grid(axis='y', alpha=0.3, linestyle='--')

                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

                figures.append(fig)

        frames = create_animation_frames_v2(figures)

        return {
            "success": True,
            "heads": heads_count,
            "tails": tails_count,
            "heads_freq": heads_count / request.n_coin,
            "tails_freq": tails_count / request.n_coin,
            "frames": frames,
            "total_frames": len(frames)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")

@app.post("/api/chapter1/needle")
async def buffon_needle(request: NeedleRequest):
    crosses = 0
    for _ in range(request.n_needle):
        y = np.random.uniform(0, request.D / 2)
        theta = np.random.uniform(0, np.pi / 2)
        if y <= (request.L / 2) * np.sin(theta):
            crosses += 1

    if crosses == 0:
        raise HTTPException(status_code=400, detail="没有发生相交情况，请增加投针次数或调整参数")

    pi_estimate = (2 * request.L * request.n_needle) / (request.D * crosses)

    x = np.linspace(0, request.n_needle, request.n_needle)
    y_est = [(2 * request.L * i) / (request.D * max(1, c)) for i, c in enumerate(range(1, request.n_needle + 1), 1)]

    fig, ax = plt.subplots()
    ax.plot(x, y_est, label="估计值")
    ax.axhline(y=np.pi, color='r', linestyle='--', label="真实值")
    ax.set_title("π值估计收敛过程")
    ax.set_xlabel("投针次数")
    ax.set_ylabel("π估计值")
    ax.legend()
    img = create_plot(fig)

    return {
        "crosses": crosses,
        "pi_estimate": pi_estimate,
        "error_rate": abs(pi_estimate - np.pi) / np.pi,
        "image": img
    }


@app.post("/api/chapter1/needle-animation")
async def buffon_needle_animation(request: NeedleRequest):
    """布丰投针动画 - 返回帧序列"""
    try:
        n_display_frames = min(30, request.n_needle // 10)
        frame_interval = max(request.n_needle // n_display_frames, 1)

        crosses = 0
        pi_estimates = []
        figures = []

        for i in range(1, request.n_needle + 1):
            y = np.random.uniform(0, request.D / 2)
            theta = np.random.uniform(0, np.pi / 2)

            if y <= (request.L / 2) * np.sin(theta):
                crosses += 1

            if i % frame_interval == 0 or i == request.n_needle:
                pi_est = (2 * request.L * i) / (request.D * max(1, crosses))
                pi_estimates.append(pi_est)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                ax1.set_xlim(0, request.D)
                ax1.set_ylim(0, request.D / 2)
                for _ in range(min(i, 100)):
                    y_rand = np.random.uniform(0, request.D / 2)
                    theta_rand = np.random.uniform(0, np.pi / 2)
                    x_start = np.random.uniform(0, request.D)
                    x_end = x_start + (request.L / 2) * np.cos(theta_rand)
                    y_end = y_rand - (request.L / 2) * np.sin(theta_rand)
                    crosses_line = y_end < 0
                    ax1.plot([x_start, x_end], [y_rand, max(0, y_end)],
                             'r-' if crosses_line else 'b-', alpha=0.3, linewidth=0.5)

                for j in range(-1, 3):
                    ax1.axhline(y=j * request.D, color='gray', linestyle='--', alpha=0.5)

                ax1.set_title(f'布丰投针模拟 (第 {i} 次)', fontsize=11, fontweight='bold')
                ax1.set_xlabel('X坐标')
                ax1.set_ylabel('Y坐标')
                ax1.grid(True, alpha=0.3)

                x_vals = list(range(1, len(pi_estimates) + 1))
                ax2.plot(x_vals, pi_estimates, 'b-', linewidth=2, label='π估计值')
                ax2.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label=f'真实值 π={np.pi:.4f}')
                ax2.set_title(f'π值收敛过程\n当前估计: {pi_est:.4f}', fontsize=11, fontweight='bold')
                ax2.set_xlabel('投针批次')
                ax2.set_ylabel('π估计值')
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)

                fig.tight_layout()
                figures.append(fig)

        frames = create_animation_frames_v2(figures)
        pi_estimate = (2 * request.L * request.n_needle) / (request.D * max(1, crosses))

        return {
            "success": True,
            "crosses": crosses,
            "pi_estimate": pi_estimate,
            "error_rate": abs(pi_estimate - np.pi) / np.pi,
            "frames": frames,
            "total_frames": len(frames)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")

@app.post("/api/chapter1/event-relation")
async def event_relation(request: EventRelationRequest):
    union_prob = request.probA + request.probB - request.probAB
    diff_prob = max(0, request.probA - request.probB)
    cond_prob = request.probAB / request.probA if request.probA > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.set_title(f"事件A∪B的概率: {union_prob:.2f}")
    circle1 = plt.Circle((0.3, 0.5), 0.3, alpha=0.5, color='blue')
    circle2 = plt.Circle((0.7, 0.5), 0.3, alpha=0.5, color='red')
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)
    ax1.text(0.1, 0.5, "A", fontsize=12)
    ax1.text(0.9, 0.5, "B", fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    ax2.set_title(f"事件A-B的概率: {diff_prob:.2f}")
    circle1 = plt.Circle((0.3, 0.5), 0.3, alpha=0.5, color='blue')
    circle2 = plt.Circle((0.7, 0.5), 0.3, alpha=0.5, color='white', edgecolor='red')
    ax2.add_patch(circle1)
    ax2.add_patch(circle2)
    ax2.text(0.1, 0.5, "A", fontsize=12)
    ax2.text(0.9, 0.5, "B", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    img = create_plot(fig)

    return {
        "probA": request.probA,
        "probB": request.probB,
        "probAB": request.probAB,
        "union_prob": union_prob,
        "diff_prob": diff_prob,
        "cond_prob": cond_prob,
        "image": img
    }


@app.post("/api/chapter1/dice-roll")
async def dice_roll(request: DiceRollRequest):
    results = [np.random.randint(1, 7) for _ in range(request.n_dice)]
    counts = [results.count(i) for i in range(1, 7)]

    fig, ax = plt.subplots()
    ax.bar(range(1, 7), counts, color='skyblue')
    ax.set_title("骰子投掷结果分布")
    ax.set_xlabel("骰子点数")
    ax.set_ylabel("出现次数")
    img = create_plot(fig)

    return {
        "counts": counts,
        "frequencies": [c / request.n_dice for c in counts],
        "image": img
    }


@app.post("/api/chapter1/geometric")
async def geometric_prob(request: GeometricProbRequest):
    prob = request.a * request.b

    fig, ax = plt.subplots()
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor='black'))
    ax.add_patch(Rectangle((0, 0), request.a, request.b, fill=True, color='blue', alpha=0.5))
    ax.text(request.a / 2, request.b / 2, f"P = {prob:.2f}", ha='center', va='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title("几何概型面积演示")
    img = create_plot(fig)

    return {
        "probability": prob,
        "image": img
    }


@app.post("/api/chapter1/conditional-prob")
async def conditional_prob(request: ConditionalProbRequest):
    pBA = request.pAB / request.pA
    pAB_indep = request.pA * request.pB
    is_independent = abs(request.pAB - pAB_indep) < 0.01
    union_prob = request.pA + request.pB - request.pAB

    fig, ax = plt.subplots()
    labels = ["P(A)", "P(B)", "P(A∩B)"]
    values = [request.pA, request.pB, request.pAB]
    ax.bar(labels, values, color=['blue', 'green', 'red'])
    ax.set_title("概率分布")
    ax.set_ylim(0, 1)
    img = create_plot(fig)

    return {
        "pBA": pBA,
        "pAB_indep": pAB_indep,
        "is_independent": is_independent,
        "union_prob": union_prob,
        "image": img
    }


@app.post("/api/chapter2/distribution")
async def distribution_plot(request: DistributionRequest):
    dist_type = request.dist_type
    params = request.params

    if dist_type == "两点分布":
        p = params.get('p', 0.5)
        x = [0, 1]
        probs = [1 - p, p]
        fig, ax = plt.subplots()
        ax.bar(x, probs, color='skyblue')
        ax.set_title(f"两点分布 PMF (p = {p:.2f})")
        ax.set_xlabel("X")
        ax.set_ylabel("概率")
        ax.set_xticks(x)
        ax.set_ylim(0, 1)

    elif dist_type == "二项分布":
        n = int(params.get('n', 10))
        p = params.get('p', 0.5)
        k = np.arange(0, n + 1)
        cdf = np.cumsum(stats.binom.pmf(k, n, p))
        fig, ax = plt.subplots()
        ax.step(k, cdf, where='post')
        ax.set_title(f"二项分布CDF (n = {n}, p = {p:.2f})")
        ax.set_xlabel("成功次数k")
        ax.set_ylabel("累积概率")
        ax.set_xticks(k)
        ax.set_ylim(0, 1.1)

    elif dist_type == "泊松分布":
        lam = params.get('lam', 5.0)
        x = np.arange(0, 31)
        probs = stats.poisson.pmf(x, lam)
        fig, ax = plt.subplots()
        ax.bar(x, probs, color='skyblue')
        ax.set_title(f"泊松分布 PMF (λ = {lam:.2f})")
        ax.set_xlabel("事件发生次数")
        ax.set_ylabel("概率")

    elif dist_type == "正态分布":
        mean = params.get('mean', 0.0)
        std = params.get('std', 1.0)
        x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
        y = stats.norm.pdf(x, mean, std)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"正态分布 PDF (μ = {mean}, σ = {std})")
        ax.set_xlabel("X")
        ax.set_ylabel("概率密度")

    elif dist_type == "均匀分布":
        min_val = params.get('min_val', 0.0)
        max_val = params.get('max_val', 1.0)
        x = np.linspace(min_val - 1, max_val + 1, 100)
        y = stats.uniform.pdf(x, loc=min_val, scale=max_val - min_val)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"均匀分布 PDF (min = {min_val}, max = {max_val})")
        ax.set_xlabel("X")
        ax.set_ylabel("概率密度")

    elif dist_type == "指数分布":
        rate = params.get('rate', 1.0)
        x = np.linspace(0, 5, 100)
        y = stats.expon.pdf(x, scale=1 / rate)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"指数分布 PDF (λ = {rate:.2f})")
        ax.set_xlabel("X")
        ax.set_ylabel("概率密度")

    else:
        raise HTTPException(status_code=400, detail="不支持的分布类型")

    img = create_plot(fig)
    return {"image": img}


@app.post("/api/chapter2/approximation")
async def distribution_approximation(request: Dict[str, Any]):
    approx_type = request.get('type', 'poisson')

    if approx_type == 'poisson':
        n = int(request.get('n', 100))
        p = request.get('p', 0.05)
        lam = n * p
        x = np.arange(0, min(n, int(lam * 3) + 1))
        binom_probs = stats.binom.pmf(x, n, p)
        poisson_probs = stats.poisson.pmf(x, lam)

        fig, ax = plt.subplots()
        ax.plot(x, binom_probs, 'b-', label="二项分布")
        ax.plot(x, poisson_probs, 'r--', label="泊松分布")
        ax.set_title(f"二项分布与泊松分布比较 (λ = {lam:.2f})")
        ax.set_xlabel("成功次数k")
        ax.set_ylabel("概率")
        ax.legend()

    else:
        n = int(request.get('n', 50))
        p = request.get('p', 0.5)
        mean = n * p
        std = np.sqrt(n * p * (1 - p))
        x = np.arange(0, n + 1)
        binom_probs = stats.binom.pmf(x, n, p)
        normal_probs = stats.norm.pdf(x, mean, std)

        fig, ax = plt.subplots()
        ax.bar(x, binom_probs, alpha=0.6, label="二项分布")
        ax.plot(x, normal_probs, 'r-', label="正态分布")
        ax.set_title(f"二项分布与正态分布比较 (n = {n}, p = {p:.2f})")
        ax.set_xlabel("成功次数k")
        ax.set_ylabel("概率")
        ax.legend()

    img = create_plot(fig)
    return {"image": img}


@app.post("/api/chapter3/joint-discrete")
async def joint_discrete(request: JointDiscreteRequest):
    total = request.prob00 + request.prob01 + request.prob10 + request.prob11
    p00 = request.prob00 / total
    p01 = request.prob01 / total
    p10 = request.prob10 / total
    p11 = request.prob11 / total

    marginal_x_0 = p00 + p01
    marginal_x_1 = p10 + p11
    marginal_y_0 = p00 + p10
    marginal_y_1 = p01 + p11

    cond_prob_y1_x0 = p01 / (p00 + p01) if (p00 + p01) > 0 else 0
    cond_prob_y0_x1 = p10 / (p10 + p11) if (p10 + p11) > 0 else 0

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    heat_data = np.array([[p00, p01], [p10, p11]])
    im = ax1.imshow(heat_data, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{heat_data[i, j]:.2f}", ha="center", va="center", color="black")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Y=0", "Y=1"])
    ax1.set_yticklabels(["X=0", "X=1"])
    ax1.set_title("联合分布 P(X,Y)")
    fig1.colorbar(im)
    img1 = create_plot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar([0, 1], [marginal_x_0, marginal_x_1], color="skyblue")
    ax2.set_title("X的边缘分布")
    ax2.set_xticks([0, 1])
    ax2.set_ylim(0, 1)
    img2 = create_plot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.bar([0, 1], [marginal_y_0, marginal_y_1], color="lightgreen")
    ax3.set_title("Y的边缘分布")
    ax3.set_xticks([0, 1])
    ax3.set_ylim(0, 1)
    img3 = create_plot(fig3)

    return {
        "joint_probs": [[p00, p01], [p10, p11]],
        "marginal_x": [marginal_x_0, marginal_x_1],
        "marginal_y": [marginal_y_0, marginal_y_1],
        "cond_prob_y1_x0": cond_prob_y1_x0,
        "cond_prob_y0_x1": cond_prob_y0_x1,
        "joint_image": img1,
        "marginal_x_image": img2,
        "marginal_y_image": img3
    }


@app.post("/api/chapter3/joint-continuous")
async def joint_continuous(request: JointContinuousRequest):
    n = 1000

    if request.dist_type_x == "正态分布":
        if request.dist_type_y == "正态分布" and request.corr_coef != 0:
            mean = [0, 0]
            cov = [[1, request.corr_coef], [request.corr_coef, 1]]
            x, y = np.random.multivariate_normal(mean, cov, n).T
        else:
            x = np.random.normal(0, 1, n)
    elif request.dist_type_x == "均匀分布":
        x = np.random.uniform(-2, 2, n)
    else:
        x = np.random.exponential(1, n)

    if request.dist_type_y == "正态分布" and not (request.dist_type_x == "正态分布" and request.corr_coef != 0):
        y = np.random.normal(0, 1, n)
    elif request.dist_type_y == "均匀分布":
        y = np.random.uniform(-2, 2, n)
    else:
        y = np.random.exponential(1, n)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(x, y, alpha=0.5)
    ax1.set_title(f"联合分布散点图 (相关系数: {request.corr_coef:.2f})")
    ax1.set_xlabel("X值")
    ax1.set_ylabel("Y值")
    img1 = create_plot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(x, bins=30, color="skyblue", density=True)
    ax2.set_title(f"X的边缘分布 ({request.dist_type_x})")
    ax2.set_xlabel("X值")
    ax2.set_ylabel("密度")
    img2 = create_plot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.hist(y, bins=30, color="lightgreen", density=True)
    ax3.set_title(f"Y的边缘分布 ({request.dist_type_y})")
    ax3.set_xlabel("Y值")
    ax3.set_ylabel("密度")
    img3 = create_plot(fig3)

    sample_corr = np.corrcoef(x, y)[0, 1] if n > 1 else 0

    return {
        "sample_corr": sample_corr,
        "scatter_image": img1,
        "marginal_x_image": img2,
        "marginal_y_image": img3
    }


@app.post("/api/chapter4/digital-features")
async def digital_features(request: DistributionRequest):
    dist_type = request.dist_type
    params = request.params
    n_samples = request.n_samples

    if dist_type == "正态分布":
        mean = params.get('mean', 0.0)
        std = params.get('std', 1.0)
        data = np.random.normal(mean, std, n_samples)
        theoretical_mean = mean
        theoretical_var = std ** 2
        theoretical_std = std
        theoretical_skew = 0
        theoretical_kurt = 0

    elif dist_type == "均匀分布":
        min_val = params.get('min_val', 0.0)
        max_val = params.get('max_val', 1.0)
        data = np.random.uniform(min_val, max_val, n_samples)
        theoretical_mean = (min_val + max_val) / 2
        theoretical_var = (max_val - min_val) ** 2 / 12
        theoretical_std = np.sqrt(theoretical_var)
        theoretical_skew = 0
        theoretical_kurt = -1.2

    elif dist_type == "泊松分布":
        lam = params.get('lam', 5.0)
        data = np.random.poisson(lam, n_samples)
        theoretical_mean = lam
        theoretical_var = lam
        theoretical_std = np.sqrt(lam)
        theoretical_skew = 1 / np.sqrt(lam)
        theoretical_kurt = 1 / lam

    elif dist_type == "指数分布":
        rate = params.get('rate', 1.0)
        data = np.random.exponential(1 / rate, n_samples)
        theoretical_mean = 1 / rate
        theoretical_var = 1 / (rate ** 2)
        theoretical_std = 1 / rate
        theoretical_skew = 2
        theoretical_kurt = 6

    elif dist_type == "二项分布":
        n = int(params.get('n', 10))
        p = params.get('p', 0.5)
        data = np.random.binomial(n, p, n_samples)
        theoretical_mean = n * p
        theoretical_var = n * p * (1 - p)
        theoretical_std = np.sqrt(theoretical_var)
        theoretical_skew = (1 - 2 * p) / np.sqrt(n * p * (1 - p)) if (n * p * (1 - p)) > 0 else 0
        theoretical_kurt = (1 - 6 * p * (1 - p)) / (n * p * (1 - p)) if (n * p * (1 - p)) > 0 else 0
    else:
        raise HTTPException(status_code=400, detail="不支持的分布类型")

    sample_mean = np.mean(data)
    sample_var = np.var(data, ddof=1)
    sample_std = np.std(data, ddof=1)
    sample_skew = stats.skew(data)
    sample_kurt = stats.kurtosis(data)

    fig, ax = plt.subplots()
    ax.hist(data, bins=30, density=True, alpha=0.6, color='blue')
    ax.set_title(f"{dist_type} 分布")
    img = create_plot(fig)

    return {
        "theoretical": {
            "mean": theoretical_mean,
            "var": theoretical_var,
            "std": theoretical_std,
            "skew": theoretical_skew,
            "kurt": theoretical_kurt
        },
        "sample": {
            "mean": sample_mean,
            "var": sample_var,
            "std": sample_std,
            "skew": sample_skew,
            "kurt": sample_kurt
        },
        "image": img
    }


@app.post("/api/chapter5/clt")
async def central_limit_theorem(request: CLTRequest):
    sample_means = []

    for _ in range(request.n_trials):
        if request.dist_type == "均匀分布":
            sample = np.random.uniform(0, 1, request.n_samples)
        elif request.dist_type == "二项分布":
            sample = np.random.binomial(10, 0.2, request.n_samples)
        elif request.dist_type == "泊松分布":
            sample = np.random.poisson(2, request.n_samples)
        else:
            sample = np.random.exponential(1, request.n_samples)
        sample_means.append(np.mean(sample))

    mean = np.mean(sample_means)
    std = np.std(sample_means)

    fig, ax = plt.subplots()
    ax.hist(sample_means, bins=30, density=True, alpha=0.6, color='blue')
    x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
    ax.plot(x, stats.norm.pdf(x, mean, std), 'r-', linewidth=2)
    ax.set_title(f"样本均值分布 (总体: {request.dist_type}, 样本量: {request.n_samples})")
    ax.set_xlabel("样本均值")
    ax.set_ylabel("密度")
    img = create_plot(fig)

    return {
        "mean": mean,
        "std": std,
        "image": img
    }


@app.post("/api/chapter5/clt-animation")
async def clt_animation(request: CLTRequest):
    """中心极限定理动画 - 返回帧序列"""
    try:
        n_frames = 30
        sample_means_all = []
        figures = []

        for frame_idx in range(1, n_frames + 1):
            current_samples = int(frame_idx * request.n_trials / n_frames)
            sample_means = []

            for _ in range(current_samples):
                if request.dist_type == "均匀分布":
                    sample = np.random.uniform(0, 1, request.n_samples)
                elif request.dist_type == "二项分布":
                    sample = np.random.binomial(10, 0.2, request.n_samples)
                elif request.dist_type == "泊松分布":
                    sample = np.random.poisson(2, request.n_samples)
                else:
                    sample = np.random.exponential(1, request.n_samples)
                sample_means.append(np.mean(sample))

            sample_means_all.extend(sample_means)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(sample_means_all, bins=35, density=True, alpha=0.6,
                    color='#667eea', edgecolor='black', linewidth=0.5, label='样本均值分布')

            mean_val = np.mean(sample_means_all)
            std_val = np.std(sample_means_all)

            x = np.linspace(mean_val - 4 * std_val, mean_val + 4 * std_val, 200)
            ax.plot(x, stats.norm.pdf(x, mean_val, std_val),
                    'r-', linewidth=2.5, label='正态拟合')

            ax.set_title(f'中心极限定理演示\n总体: {request.dist_type}, 样本量: {request.n_samples}\n'
                         f'抽样次数: {current_samples}, 均值: {mean_val:.3f}, 标准差: {std_val:.3f}',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('样本均值', fontsize=11)
            ax.set_ylabel('密度', fontsize=11)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            figures.append(fig)

        frames = create_animation_frames_v2(figures)

        return {
            "success": True,
            "mean": np.mean(sample_means_all),
            "std": np.std(sample_means_all),
            "frames": frames,
            "total_frames": len(frames)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")

@app.post("/api/chapter5/lln")
async def law_large_numbers(request: LLNRequest):
    means = []
    errors = []

    for i in range(1, request.num_trials + 1):
        if request.dist_type == "均匀分布":
            data = np.random.uniform(0, 1, i)
            expected_mean = 0.5
        elif request.dist_type == "二项分布":
            data = np.random.binomial(10, 0.5, i)
            expected_mean = 5
        else:
            data = np.random.poisson(5, i)
            expected_mean = 5

        current_mean = np.mean(data)
        means.append(current_mean)
        errors.append(abs(current_mean - expected_mean))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('试验次数')
    ax1.set_ylabel('样本均值', color='tab:blue')
    ax1.plot(range(1, request.num_trials + 1), means, color='tab:blue')
    ax1.axhline(y=expected_mean, color='r', linestyle='--', label="期望值")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('估计误差', color='tab:red')
    ax2.plot(range(1, request.num_trials + 1), errors, color='tab:red', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title(f"大数定律演示 (分布: {request.dist_type})")
    img = create_plot(fig)

    return {
        "expected_mean": expected_mean,
        "final_mean": means[-1],
        "image": img
    }


@app.post("/api/chapter5/lln-animation")
async def law_large_numbers_animation(request: LLNRequest):
    """大数定律动画 - 返回帧序列"""
    try:
        n_frames = 30
        frame_interval = max(request.num_trials // n_frames, 1)

        means = []
        errors = []
        figures = []
        expected_mean = 0.5

        for i in range(1, request.num_trials + 1):
            if request.dist_type == "均匀分布":
                data_point = np.random.uniform(0, 1)
                expected_mean = 0.5
            elif request.dist_type == "二项分布":
                data_point = np.random.binomial(10, 0.5)
                expected_mean = 5
            else:
                data_point = np.random.poisson(5)
                expected_mean = 5

            means.append((means[-1] * (i - 1) + data_point) / i if i > 1 else data_point)
            errors.append(abs(means[-1] - expected_mean))

            if i % frame_interval == 0 or i == request.num_trials:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

                x_vals = list(range(1, len(means) + 1))
                ax1.plot(x_vals, means, 'b-', linewidth=2, label='样本均值')
                ax1.axhline(y=expected_mean, color='r', linestyle='--',
                            linewidth=2.5, label=f'期望值 {expected_mean}')
                ax1.set_title(f'大数定律演示 - 样本均值收敛\n分布: {request.dist_type}',
                              fontsize=12, fontweight='bold')
                ax1.set_xlabel('试验次数', fontsize=11)
                ax1.set_ylabel('样本均值', fontsize=11)
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)

                ax2.plot(x_vals, errors, 'orange', linewidth=2, label='估计误差')
                ax2.fill_between(x_vals, errors, alpha=0.3, color='orange')
                ax2.set_title('误差变化', fontsize=11, fontweight='bold')
                ax2.set_xlabel('试验次数', fontsize=11)
                ax2.set_ylabel('绝对误差', fontsize=11)
                ax2.legend(loc='best')
                ax2.grid(True, alpha=0.3)

                fig.tight_layout()
                figures.append(fig)

        frames = create_animation_frames_v2(figures)

        return {
            "success": True,
            "expected_mean": expected_mean,
            "final_mean": means[-1],
            "frames": frames,
            "total_frames": len(frames)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")

@app.post("/api/chapter6/sampling-dist")
async def sampling_distribution(request: SamplingDistRequest):
    num_simulations = 1000
    sample_means = []
    sample_vars = []

    for _ in range(num_simulations):
        sample = np.random.normal(request.mu, request.sigma, request.n)
        sample_means.append(np.mean(sample))
        sample_vars.append(np.var(sample, ddof=1))

    fig1, ax1 = plt.subplots()
    ax1.hist(sample_means, bins=30, alpha=0.7, color='blue')
    ax1.axvline(x=np.mean(sample_means), color='red', linestyle='dashed',
                label=f"均值: {np.mean(sample_means):.4f}")
    ax1.set_title("样本均值的分布")
    ax1.set_xlabel("样本均值")
    ax1.set_ylabel("频率")
    ax1.legend()
    img1 = create_plot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(sample_vars, bins=30, alpha=0.7, color='green')
    ax2.axvline(x=np.mean(sample_vars), color='red', linestyle='dashed',
                label=f"均值: {np.mean(sample_vars):.4f}")
    ax2.set_title("样本方差的分布")
    ax2.set_xlabel("样本方差")
    ax2.set_ylabel("频率")
    ax2.legend()
    img2 = create_plot(fig2)

    return {
        "mean_expectation": {
            "theoretical": request.mu,
            "actual": np.mean(sample_means)
        },
        "mean_variance": {
            "theoretical": request.sigma ** 2 / request.n,
            "actual": np.var(sample_means)
        },
        "var_expectation": {
            "theoretical": request.sigma ** 2,
            "actual": np.mean(sample_vars)
        },
        "mean_image": img1,
        "var_image": img2
    }


@app.post("/api/chapter6/order-stats")
async def order_statistics(request: OrderStatsRequest):
    num_samples = 1000
    mins = []
    maxs = []
    medians = []

    for _ in range(num_samples):
        if request.dist_type == "正态分布":
            sample = np.random.normal(0, 1, request.sample_size)
        elif request.dist_type == "均匀分布":
            sample = np.random.uniform(0, 1, request.sample_size)
        else:
            sample = np.random.exponential(1, request.sample_size)

        ordered = np.sort(sample)
        mins.append(ordered[0])
        maxs.append(ordered[-1])
        medians.append(np.median(ordered))

    fig1, ax1 = plt.subplots()
    ax1.hist(mins, bins=30, alpha=0.7, color='blue')
    ax1.set_title("最小值的分布")
    ax1.set_xlabel("最小值")
    ax1.set_ylabel("频率")
    img1 = create_plot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.hist(medians, bins=30, alpha=0.7, color='green')
    ax2.set_title("中位数的分布")
    ax2.set_xlabel("中位数")
    ax2.set_ylabel("频率")
    img2 = create_plot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.hist(maxs, bins=30, alpha=0.7, color='red')
    ax3.set_title("最大值的分布")
    ax3.set_xlabel("最大值")
    ax3.set_ylabel("频率")
    img3 = create_plot(fig3)

    return {
        "stats": {
            "min": {"mean": np.mean(mins), "std": np.std(mins)},
            "median": {"mean": np.mean(medians), "std": np.std(medians)},
            "max": {"mean": np.mean(maxs), "std": np.std(maxs)}
        },
        "min_image": img1,
        "median_image": img2,
        "max_image": img3
    }


@app.post("/api/chapter6/common-distributions")
async def common_distributions(request: Dict[str, Any]):
    dist_name = request.get('dist', 't')

    if dist_name == 't':
        df = request.get('df', 5)
        x = np.linspace(-4, 4, 100)
        y = stats.t.pdf(x, df=df)
        y_norm = stats.norm.pdf(x, 0, 1)
        fig, ax = plt.subplots()
        ax.plot(x, y, label=f"t分布 (df={df})")
        ax.plot(x, y_norm, 'r--', label="标准正态分布")
        ax.set_title(f"t分布与正态分布比较")
        ax.set_xlabel("x")
        ax.set_ylabel("概率密度")
        ax.legend()

    elif dist_name == 'f':
        df1 = request.get('df1', 5)
        df2 = request.get('df2', 5)
        x = np.linspace(0, 5, 100)
        y = stats.f.pdf(x, df1, df2)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(f"F分布 (df1={df1}, df2={df2})")
        ax.set_xlabel("x")
        ax.set_ylabel("概率密度")

    else:
        df = request.get('df', 5)
        x = np.linspace(0, 20, 100)
        y = stats.chi2.pdf(x, df=df)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title(f"卡方分布 (自由度 = {df})")
        ax.set_xlabel("x")
        ax.set_ylabel("概率密度")

    img = create_plot(fig)
    return {"image": img}


@app.post("/api/chapter7/moment-estimation")
async def moment_estimation(request: MomentEstimationRequest):
    a_true = 0
    b_true = 1
    data = np.random.uniform(a_true, b_true, request.n)

    mu1 = np.mean(data)
    mu2 = np.mean(data ** 2)
    a_hat = mu1 - np.sqrt(3 * (mu2 - mu1 ** 2))
    b_hat = mu1 + np.sqrt(3 * (mu2 - mu1 ** 2))

    fig, ax = plt.subplots()
    ax.hist(data, bins=30, alpha=0.6, color='blue')
    ax.axvline(x=a_hat, color='red', linestyle='dashed', label=f"a估计值: {a_hat:.4f}")
    ax.axvline(x=b_hat, color='green', linestyle='dashed', label=f"b估计值: {b_hat:.4f}")
    ax.axvline(x=a_true, color='black', linestyle='-', alpha=0.3, label=f"真实a: {a_true}")
    ax.axvline(x=b_true, color='black', linestyle='-', alpha=0.3, label=f"真实b: {b_true}")
    ax.set_title(f"均匀分布矩估计 (样本量: {request.n})")
    ax.set_xlabel("样本值")
    ax.set_ylabel("频率")
    ax.legend()
    img = create_plot(fig)

    return {
        "true_values": {"a": a_true, "b": b_true},
        "estimates": {"a": a_hat, "b": b_hat},
        "errors": {"a": abs(a_hat - a_true), "b": abs(b_hat - b_true)},
        "image": img
    }


@app.post("/api/chapter7/mle")
async def maximum_likelihood(request: MLEstimationRequest):
    mu_true = 5
    sigma_true = 2
    data = np.random.normal(mu_true, sigma_true, request.n)

    mu_hat = np.mean(data)
    sigma_hat_mle = np.sqrt(np.mean((data - mu_hat) ** 2))
    sigma_hat_unbiased = np.sqrt(np.var(data, ddof=1))

    fig, ax = plt.subplots()
    ax.hist(data, bins=30, density=True, alpha=0.6, color='blue')
    x = np.linspace(mu_hat - 3 * sigma_hat_mle, mu_hat + 3 * sigma_hat_mle, 100)
    ax.plot(x, stats.norm.pdf(x, mu_hat, sigma_hat_mle), 'r-',
            label=f"MLE: N({mu_hat:.2f}, {sigma_hat_mle:.2f})")
    ax.plot(x, stats.norm.pdf(x, mu_true, sigma_true), 'g--', label=f"真实: N({mu_true}, {sigma_true})")
    ax.set_title(f"正态分布最大似然估计 (样本量: {request.n})")
    ax.set_xlabel("样本值")
    ax.set_ylabel("密度")
    ax.legend()
    img = create_plot(fig)

    return {
        "true_values": {"mu": mu_true, "sigma": sigma_true},
        "estimates": {
            "mu": mu_hat,
            "sigma_mle": sigma_hat_mle,
            "sigma_unbiased": sigma_hat_unbiased
        },
        "errors": {
            "mu": abs(mu_hat - mu_true),
            "sigma_mle": abs(sigma_hat_mle - sigma_true),
            "sigma_unbiased": abs(sigma_hat_unbiased - sigma_true)
        },
        "image": img
    }


@app.post("/api/chapter7/efficiency")
async def estimator_efficiency(request: EstimatorEfficiencyRequest):
    try:
        num_samples = 500
        estimates = []
        crlb_values = []

        for _ in range(num_samples):
            if request.dist_type == "正态分布":
                sample = np.random.normal(5, 2, request.sample_size)
                estimates.append(np.mean(sample))
                crlb = 4 / request.sample_size
                crlb_values.append(crlb)
            elif request.dist_type == "均匀分布":
                sample = np.random.uniform(0, 10, request.sample_size)
                estimates.append(np.mean(sample))
                crlb = (10 ** 2 / 12) / request.sample_size
                crlb_values.append(crlb)
            elif request.dist_type == "泊松分布":
                sample = np.random.poisson(5, request.sample_size)
                estimates.append(np.mean(sample))
                crlb = 5 / request.sample_size
                crlb_values.append(crlb)
            else:
                raise HTTPException(status_code=400, detail=f"不支持的分布类型: {request.dist_type}")

        var_estimate = np.var(estimates)
        avg_crlb = np.mean(crlb_values)

        fig, ax = plt.subplots()
        ax.hist(estimates, bins=30, alpha=0.6, color='blue')
        ax.set_title(f"{request.dist_type} 估计量分布")
        ax.set_xlabel("估计值")
        ax.set_ylabel("频率")
        img = create_plot(fig)

        return {
            "variance": float(var_estimate),
            "crlb": float(avg_crlb),
            "is_efficient": bool(var_estimate < avg_crlb * 1.05),
            "image": img
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")



# 将整个函数替换为:
@app.post("/api/chapter8/confidence-interval")
async def confidence_interval(request: ConfidenceIntervalRequest):
    try:
        population_mean = 50
        population_std = 10
        sample = np.random.normal(population_mean, population_std, request.sample_size)

        sample_mean = float(np.mean(sample))
        sample_std = float(np.std(sample, ddof=1))

        alpha = 1 - request.confidence_level
        t_value = float(stats.t.ppf(1 - alpha / 2, df=request.sample_size - 1))
        margin_error = t_value * sample_std / np.sqrt(request.sample_size)
        ci_lower = sample_mean - margin_error
        ci_upper = sample_mean + margin_error

        fig, ax = plt.subplots()
        ax.hist(sample, bins=30, alpha=0.6, color='blue')
        ax.axvline(x=sample_mean, color='red', linestyle='-', label=f"样本均值: {sample_mean:.2f}")
        ax.axvline(x=ci_lower, color='green', linestyle='--',
                   label=f"{request.confidence_level * 100}% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
        ax.axvline(x=ci_upper, color='green', linestyle='--')
        ax.axvline(x=population_mean, color='black', linestyle='-.', label=f"总体均值: {population_mean}")
        ax.set_title("样本数据分布")
        ax.set_xlabel("样本值")
        ax.set_ylabel("频率")
        ax.legend()
        img = create_plot(fig)

        return {
            "sample_mean": sample_mean,
            "sample_std": sample_std,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "margin_error": float(margin_error),
            "contains_true_mean": bool(ci_lower <= population_mean <= ci_upper),
            "image": img
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")




@app.post("/api/chapter8/normal-ci")
async def normal_confidence_interval(request: Dict[str, Any]):
    pop_mean = request.get('pop_mean', 0.0)
    pop_std = request.get('pop_std', 1.0)
    sample_size = request.get('sample_size', 50)
    confidence_level = request.get('confidence_level', 0.95)

    sample = np.random.normal(pop_mean, pop_std, sample_size)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)

    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha / 2, df=sample_size - 1)
    margin_error_mean = t_value * sample_std / np.sqrt(sample_size)
    ci_mean = [sample_mean - margin_error_mean, sample_mean + margin_error_mean]

    chi2_lower = stats.chi2.ppf(alpha / 2, df=sample_size - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=sample_size - 1)
    ci_var = [
        (sample_size - 1) * sample_std ** 2 / chi2_upper,
        (sample_size - 1) * sample_std ** 2 / chi2_lower
    ]

    fig1, ax1 = plt.subplots()
    ax1.errorbar(x=0, y=sample_mean, yerr=margin_error_mean, fmt='bo', capsize=5, label="样本均值")
    ax1.axhline(y=pop_mean, color='r', linestyle='--', label="真实均值")
    ax1.set_title(f"均值的{confidence_level * 100}%置信区间")
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_xticks([])
    ax1.legend()
    img1 = create_plot(fig1)

    fig2, ax2 = plt.subplots()
    yerr = [[sample_std ** 2 - ci_var[0]], [ci_var[1] - sample_std ** 2]]
    ax2.errorbar(x=0, y=sample_std ** 2, yerr=yerr, fmt='go', capsize=5, label="样本方差")
    ax2.axhline(y=pop_std ** 2, color='r', linestyle='--', label="真实方差")
    ax2.set_title(f"方差的{confidence_level * 100}%置信区间")
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_xticks([])
    ax2.legend()
    img2 = create_plot(fig2)

    return {
        "mean_ci": ci_mean,
        "var_ci": ci_var,
        "contains_mean": ci_mean[0] <= pop_mean <= ci_mean[1],
        "contains_var": ci_var[0] <= pop_std ** 2 <= ci_var[1],
        "mean_image": img1,
        "var_image": img2
    }


@app.post("/api/chapter9/one-sample-ttest")
async def one_sample_ttest(request: TTestRequest):
    population_std = 2.0
    sample = np.random.normal(request.true_mean, population_std, request.sample_size)

    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)

    t_stat, p_value = stats.ttest_1samp(sample, request.hypothesized_mean)
    df = request.sample_size - 1

    t_critical = stats.t.ppf(1 - request.alpha / 2, df)
    reject_low = request.hypothesized_mean - t_critical * (sample_std / np.sqrt(request.sample_size))
    reject_high = request.hypothesized_mean + t_critical * (sample_std / np.sqrt(request.sample_size))

    decision = "拒绝原假设" if p_value < request.alpha else "不拒绝原假设"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sample, bins=30, alpha=0.6, color='blue', density=True, label="样本分布")
    x = np.linspace(min(sample), max(sample), 100)
    ax.plot(x, stats.norm.pdf(x, sample_mean, sample_std), 'r-', label="样本正态近似")
    ax.axvline(x=request.hypothesized_mean, color='green', linestyle='--',
               label=f"假设均值 μ₀={request.hypothesized_mean}")
    ax.axvline(x=request.true_mean, color='black', linestyle='-.', label=f"真实均值 μ={request.true_mean}")
    ax.axvline(x=sample_mean, color='red', linestyle='-', label=f"样本均值={sample_mean:.2f}")
    ax.axvspan(min(sample), reject_low, color='gray', alpha=0.3, label="拒绝域")
    ax.axvspan(reject_high, max(sample), color='gray', alpha=0.3)
    ax.set_title(f"单样本t检验结果 (α={request.alpha})")
    ax.set_xlabel("样本值")
    ax.set_ylabel("密度")
    ax.legend()
    img = create_plot(fig)

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "df": df,
        "t_critical": t_critical,
        "decision": decision,
        "sample_mean": sample_mean,
        "image": img
    }


@app.post("/api/chapter9/two-sample-ttest")
async def two_sample_ttest(request: TwoSampleTTestRequest):
    sample1 = np.random.normal(request.mean1, request.std1, request.sample_size1)
    sample2 = np.random.normal(request.mean2, request.std2, request.sample_size2)

    mean1_sample = np.mean(sample1)
    mean2_sample = np.mean(sample2)

    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=request.equal_var)
    decision = "拒绝原假设" if p_value < request.alpha else "不拒绝原假设"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sample1, bins=30, alpha=0.5, color='blue', density=True, label="样本1")
    ax.hist(sample2, bins=30, alpha=0.5, color='green', density=True, label="样本2")
    ax.axvline(x=mean1_sample, color='blue', linestyle='-', label=f"样本1均值={mean1_sample:.2f}")
    ax.axvline(x=mean2_sample, color='green', linestyle='-', label=f"样本2均值={mean2_sample:.2f}")
    ax.set_title("两样本分布比较")
    ax.set_xlabel("样本值")
    ax.set_ylabel("密度")
    ax.legend()
    img = create_plot(fig)

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_diff": mean1_sample - mean2_sample,
        "decision": decision,
        "mean1": mean1_sample,
        "mean2": mean2_sample,
        "image": img
    }


@app.post("/api/chapter9/t-test-animation")
async def t_test_animation(request: TTestRequest):
    """t检验动画 - 返回帧序列"""
    try:
        n_frames = 25
        sample = np.random.normal(request.true_mean, 1, request.sample_size)
        figures = []

        for frame_idx in range(1, n_frames + 1):
            current_size = max(5, int(frame_idx * request.sample_size / n_frames))
            current_sample = sample[:current_size]

            sample_mean = np.mean(current_sample)
            sample_std = np.std(current_sample, ddof=1)
            t_stat = (sample_mean - request.hypothesized_mean) / (sample_std / np.sqrt(current_size))

            x = np.linspace(-4, 4, 300)
            df = current_size - 1
            t_dist = stats.t.pdf(x, df)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, t_dist, 'b-', linewidth=2.5, label=f't分布 (df={df})')

            ax.fill_between(x[x <= -abs(t_stat)], t_dist[x <= -abs(t_stat)],
                            color='red', alpha=0.3, label='拒绝域')
            ax.fill_between(x[x >= abs(t_stat)], t_dist[x >= abs(t_stat)],
                            color='red', alpha=0.3)

            ax.axvline(x=t_stat, color='green', linestyle='--',
                       linewidth=2.5, label=f't统计量 = {t_stat:.3f}')
            ax.axvline(x=request.hypothesized_mean, color='orange',
                       linestyle=':', linewidth=2, label=f'H₀: μ={request.hypothesized_mean}')

            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

            ax.set_title(f'单样本t检验动态演示\n样本量: {current_size}, t统计量: {t_stat:.3f}, '
                         f'p值: {p_value:.4f}\n{"拒绝H₀" if p_value < request.alpha else "不拒绝H₀"} '
                         f'(α={request.alpha})',
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('t值', fontsize=11)
            ax.set_ylabel('概率密度', fontsize=11)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(0, max(t_dist) * 1.1)
            figures.append(fig)

        frames = create_animation_frames_v2(figures)

        sample_mean_full = np.mean(sample)
        sample_std_full = np.std(sample, ddof=1)
        t_stat_full = (sample_mean_full - request.hypothesized_mean) / (sample_std_full / np.sqrt(request.sample_size))
        p_value_full = 2 * (1 - stats.t.cdf(abs(t_stat_full), request.sample_size - 1))

        return {
            "success": True,
            "sample_mean": sample_mean_full,
            "sample_std": sample_std_full,
            "t_statistic": t_stat_full,
            "p_value": p_value_full,
            "reject_null": p_value_full < request.alpha,
            "frames": frames,
            "total_frames": len(frames)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")

@app.post("/api/code-runner/execute")
async def execute_user_code(request: CodeExecutionRequest):
    """执行用户提交的 Python 代码并返回生成的图表"""
    import time
    import sys
    from io import StringIO

    start_time = time.time()

    try:
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        local_vars = {}
        global_vars = {
            'np': np,
            'plt': plt,
            'pd': pd,
            'stats': stats,
            '__name__': '__main__'
        }

        exec(request.code, global_vars, local_vars)

        sys.stdout = old_stdout
        output = redirected_output.getvalue()

        figs = [manager.canvas.figure for manager in plt._pylab_helpers.Gcf.get_all_fig_managers()]

        if not figs:
            execution_time = time.time() - start_time
            return {
                "success": True,
                "message": "代码执行成功，但未生成图表",
                "output": output,
                "execution_time": execution_time
            }

        last_fig = figs[-1]
        img_base64 = create_plot(last_fig)

        plt.close('all')

        execution_time = time.time() - start_time

        return {
            "success": True,
            "image": img_base64,
            "output": output,
            "execution_time": execution_time
        }

    except Exception as e:
        execution_time = time.time() - start_time
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"代码执行错误:\n{str(e)}"
        )


@app.post("/api/chapter9/t-test")
async def t_test_alias(request: TTestRequest):
    """单样本t检验的别名路由"""
    return await one_sample_ttest(request)


@app.post("/api/chapter9/two-sample-t-test")
async def two_sample_t_test_alias(request: TwoSampleTTestRequest):
    """双样本t检验的别名路由"""
    return await two_sample_ttest(request)



@app.post("/api/ai/chat")
async def ai_chat(request: AIMessage):
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        ai_provider = os.getenv("AI_PROVIDER", "openai").lower()

        if not api_key or api_key == "your_api_key_here":
            return {
                "response": "🤖 AI 教学助手\n\n"
                            "当前未配置 API 密钥，AI 功能暂不可用。\n\n"
                            "💡 配置方法：\n\n"
                            "【使用通义千问（推荐）】\n"
                            "1. 访问 https://dashscope.console.aliyun.com\n"
                            "2. 注册并创建 API-KEY\n"
                            "3. 在 .env 文件中设置：\n"
                            "   AI_PROVIDER=dashscope\n"
                            "   OPENAI_API_KEY=你的通义千问API-Key\n\n"
                            "📚 不过没关系！你仍然可以使用平台的所有概率实验功能！",
                "success": False
            }

        system_prompt = """你是一个专业的概率论与数理统计教学助手。你的任务是帮助学生理解概率统计概念、解释实验结果、回答相关问题。请用简洁易懂的语言回答，必要时可以给出数学公式和示例。"""

        user_message = request.message
        if request.context:
            user_message = f"当前学习内容：{request.context}\n\n问题：{user_message}"

        if ai_provider == "dashscope":
            if not DASHSCOPE_AVAILABLE:
                return {
                    "response": "dashscope库未安装。请运行 pip install dashscope 安装通义千问SDK。",
                    "success": False
                }
            return await call_dashscope(system_prompt, user_message, api_key)
        else:
            if not OPENAI_AVAILABLE:
                return {
                    "response": "OpenAI库未安装。请运行 pip install openai 安装。",
                    "success": False
                }
            return await call_openai(system_prompt, user_message, api_key)

    except Exception as e:
        return {
            "response": f"AI服务出错：{str(e)}",
            "success": False
        }


async def call_openai(system_prompt, user_message, api_key):
    """调用 OpenAI API"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )

        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
        )

        return {
            "response": response.choices[0].message.content,
            "success": True
        }
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return {
                "response": "API密钥验证失败。请检查OPENAI_API_KEY是否正确配置。",
                "success": False
            }
        elif "rate limit" in error_msg.lower():
            return {
                "response": "API调用频率超限。请稍后再试。",
                "success": False
            }
        else:
            return {
                "response": f"OpenAI服务出错：{error_msg}",
                "success": False
            }


async def call_dashscope(system_prompt, user_message, api_key):
    """调用阿里云通义千问 API"""
    try:
        from http import HTTPStatus

        dashscope.api_key = api_key

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        import asyncio
        loop = asyncio.get_event_loop()

        def call_api():
            response = dashscope.Generation.call(
                model=dashscope.Generation.Models.qwen_turbo,
                messages=messages,
                result_format='message'
            )
            return response

        response = await loop.run_in_executor(None, call_api)

        if response.status_code == HTTPStatus.OK:
            return {
                "response": response.output.choices[0].message.content,
                "success": True
            }
        else:
            return {
                "response": f"通义千问API调用失败：{response.code} - {response.message}",
                "success": False
            }

    except Exception as e:
        error_msg = str(e)
        if "ModuleNotFoundError" in error_msg or "No module named 'dashscope'" in error_msg:
            return {
                "response": "dashscope库未安装。请运行 pip install dashscope 安装。",
                "success": False
            }
        else:
            return {
                "response": f"通义千问服务出错：{error_msg}",
                "success": False
            }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    uvicorn.run("main:app", host=host, port=port, reload=debug)


@app.post("/api/chapter2/poisson-approx-animation")
async def poisson_approx_animation(request: PoissonApproxRequest):
    """泊松近似二项分布动画 - 返回帧序列"""
    try:
        n = request.n
        p = request.p
        lam = n * p
        figures = []

        # 步骤1：二项分布展开（10帧）
        for frame in range(1, 11):
            progress = frame / 10
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(0, int(n * progress) + 1)
            if len(x) > 0:
                y = stats.binom.pmf(x, n, p)
                ax.bar(x, y, color='steelblue', width=0.4, alpha=0.8, edgecolor='white')
            ax.set_title(f"二项分布 B(n={n}, p={p}) 正在展开... ({frame}/10)")
            ax.set_xlabel("成功次数 k")
            ax.set_ylabel("概率")
            ax.grid(alpha=0.3)
            figures.append(fig)

        # 步骤2：泊松分布淡入叠加（15帧）
        x_max = min(int(lam + 4 * np.sqrt(lam)) + 1, n, 200)
        x_pois = np.arange(0, x_max + 1)
        y_pois = stats.poisson.pmf(x_pois, lam)
        x_binom = np.arange(0, n + 1)
        y_binom = stats.binom.pmf(x_binom, n, p)

        for frame in range(1, 16):
            alpha = frame / 15
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x_binom, y_binom, color='steelblue', width=0.4, alpha=0.6,
                   edgecolor='white', label='二项分布')
            ax.plot(x_pois, y_pois, 'rx--', markersize=6, alpha=alpha,
                    linewidth=1.5, label=f'泊松分布 P(λ={lam:.2f})')
            ax.set_title(f"泊松分布正在淡入... ({frame}/15)")
            ax.set_xlabel("成功次数 k")
            ax.set_ylabel("概率")
            ax.legend()
            ax.grid(alpha=0.3)
            figures.append(fig)

        # 步骤3：误差分析（10帧）
        max_err = np.max(np.abs(y_binom[:x_max+1] - y_pois))
        for frame in range(1, 11):
            alpha = frame / 10
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x_binom, y_binom, color='steelblue', width=0.4, alpha=0.5,
                   edgecolor='white', label='二项分布')
            ax.plot(x_pois, y_pois, 'rx--', markersize=6, linewidth=2,
                    label=f'泊松分布 P(λ={lam:.2f})')
            if frame > 5:
                ax.text(0.02, 0.95, f"最大误差: {max_err:.6f}",
                        transform=ax.transAxes, fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax.set_title(f"近似误差分析 ({frame}/10)")
            ax.set_xlabel("成功次数 k")
            ax.set_ylabel("概率")
            ax.legend()
            ax.grid(alpha=0.3)
            figures.append(fig)

        frames = create_animation_frames_v2(figures)
        return {
            "success": True,
            "frames": frames,
            "total_frames": len(frames),
            "n": n,
            "p": p,
            "lambda": lam,
            "max_error": float(max_err)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")



@app.post("/api/chapter2/normal-approx-animation")
async def normal_approx_animation(request: NormalApproxRequest):
    """二项分布近似正态分布动画 - 返回帧序列"""
    try:
        n = request.n
        p = request.p
        mean = n * p
        std = np.sqrt(n * p * (1 - p))
        figures = []

        # 步骤1：小n时二项分布（10帧）
        n_small = min(5, n)
        for frame in range(1, 11):
            progress = frame / 10
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(0, int(n_small * progress) + 1)
            if len(x) > 0:
                y = stats.binom.pmf(x, n_small, p)
                ax.bar(x, y, color='steelblue', width=0.4)
            ax.set_title(f"二项分布 B(n={n_small}, p={p}) 正在展开... ({frame}/10)")
            ax.set_xlabel("成功次数 k")
            ax.set_ylabel("概率")
            ax.grid(alpha=0.3)
            figures.append(fig)

        # 步骤2：n从小到大的生长动画（15帧）
        n_values = np.unique(np.linspace(1, n, 15).astype(int))
        if len(n_values) < 2:
            n_values = np.array([1, n])
        for idx, current_n in enumerate(n_values):
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(0, current_n + 1)
            y = stats.binom.pmf(x, current_n, p)
            bar_width = min(0.8, max(0.1, 20.0 / current_n))
            ax.bar(x, y, color='steelblue', width=bar_width, alpha=0.8, edgecolor='white')
            mean_now = current_n * p
            std_now = np.sqrt(current_n * p * (1 - p))
            x_min = max(0, int(mean_now - 4 * std_now))
            x_max = min(current_n, int(mean_now + 4 * std_now))
            if x_max > x_min:
                ax.set_xlim(x_min - 0.5, x_max + 0.5)
            ax.set_title(f"n 正在增大... ({idx + 1}/{len(n_values)})  当前 n = {current_n}")
            ax.set_xlabel("成功次数 k")
            ax.set_ylabel("概率")
            ax.grid(alpha=0.3)
            figures.append(fig)

        # 步骤3：正态曲线淡入叠加（15帧）
        x_norm = np.linspace(max(0, mean - 4 * std), min(n, mean + 4 * std), 200)
        y_norm = stats.norm.pdf(x_norm, mean, std)
        for frame in range(1, 16):
            alpha = frame / 15
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(0, n + 1)
            y = stats.binom.pmf(x, n, p)
            bar_width = min(0.8, max(0.1, 20.0 / n))
            ax.bar(x, y, color='steelblue', width=bar_width, alpha=0.6,
                   edgecolor='white', label='二项分布')
            ax.plot(x_norm, y_norm, 'r-', linewidth=2, alpha=alpha,
                    label=f'正态分布 N({mean:.1f}, {std:.1f}²)')
            ax.set_title(f"正态曲线正在叠加... ({frame}/15)")
            ax.set_xlabel("成功次数 k")
            ax.set_ylabel("概率 / 密度")
            ax.legend()
            ax.grid(alpha=0.3)
            figures.append(fig)

        frames = create_animation_frames_v2(figures)
        return {
            "success": True,
            "frames": frames,
            "total_frames": len(frames),
            "n": n,
            "p": p,
            "mean": float(mean),
            "std": float(std)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")



@app.post("/api/chapter3/joint-dist-animation")
async def joint_dist_animation(request):
    """联合分布可视化动画 - 返回帧序列"""
    try:
        data = await request.json()
        dist_x = data.get("dist_x", "正态分布")
        dist_y = data.get("dist_y", "正态分布")
        rho = data.get("rho", 0.0)
        n_total = data.get("n_total", 1000)

        figures = []

        # 生成相关样本
        cov = [[1, rho], [rho, 1]]
        z = np.random.multivariate_normal([0, 0], cov, n_total)
        u = stats.norm.cdf(z[:, 0])
        v = stats.norm.cdf(z[:, 1])

        def inverse_cdf(dist, w):
            if dist == "正态分布":
                return stats.norm.ppf(w, loc=0, scale=1)
            elif dist == "均匀分布":
                return stats.uniform.ppf(w, loc=-3, scale=6)
            else:
                return stats.expon.ppf(w, scale=1)

        x = inverse_cdf(dist_x, u)
        y = inverse_cdf(dist_y, v)

        # 步骤1：生成样本点（15帧）
        for frame in range(1, 16):
            n_show = int(len(x) * frame / 15)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x[:n_show], y[:n_show], alpha=0.6, s=20,
                     color="steelblue", edgecolor="white", linewidth=0.3)
            ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
            ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
            ax.set_title(f"正在生成样本点... ({n_show}/{len(x)})")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(alpha=0.3)
            figures.append(fig)

        # 步骤2：构建联合曲面（20帧）
        from scipy.stats import gaussian_kde
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        x_min, x_max = x.min() - 0.5, x.max() + 0.5
        y_min, y_max = y.min() - 0.5, y.max() + 0.5
        nx, ny = 40, 40
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(positions).T, X.shape)
        z_max = Z.max()

        for frame in range(1, 21):
            t = frame / 20
            progress = 1 - (1 - t) ** 2
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(X, Y, Z * progress, cmap="viridis",
                                   alpha=0.9, edgecolor="none", vmin=0, vmax=z_max)
            fig.colorbar(surf, shrink=0.4, aspect=8)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("联合概率密度 f(x,y)")
            ax.set_title(f"联合分布曲面正在生长... ({frame}/20)", fontsize=12)
            ax.view_init(elev=30, azim=-60)
            figures.append(fig)

        # 步骤3：投影边缘分布（15帧）
        x_marginal = np.sum(Z, axis=0) * (y_max - y_min) / ny
        y_marginal = np.sum(Z, axis=1) * (x_max - x_min) / nx
        x_marginal = x_marginal / x_marginal.max() * z_max * 0.35
        y_marginal = y_marginal / y_marginal.max() * z_max * 0.35

        for frame in range(1, 16):
            t = frame / 15
            progress = 1 - (1 - t) ** 2
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.25, edgecolor="none", vmin=0, vmax=z_max)
            z_x = x_marginal * progress
            ax.plot(x_grid, np.full_like(x_grid, y_grid.min()), np.zeros_like(x_grid), "r-", alpha=0.2, linewidth=1)
            ax.plot(x_grid, np.full_like(x_grid, y_grid.min()), z_x, "r-", linewidth=3, label="X 边缘分布")
            z_y = y_marginal * progress
            ax.plot(np.full_like(y_grid, x_grid.min()), y_grid, np.zeros_like(y_grid), "b-", alpha=0.2, linewidth=1)
            ax.plot(np.full_like(y_grid, x_grid.min()), y_grid, z_y, "b-", linewidth=3, label="Y 边缘分布")
            if frame > 5:
                ax.text2D(0.02, 0.95, "红色 = X 的边缘分布", transform=ax.transAxes, color="red", fontsize=10)
                ax.text2D(0.02, 0.90, "蓝色 = Y 的边缘分布", transform=ax.transAxes, color="blue", fontsize=10)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("概率密度")
            ax.set_title(f"边缘分布投影中... ({frame}/15)", fontsize=12)
            ax.view_init(elev=25, azim=-50)
            ax.legend(loc="upper left")
            figures.append(fig)

        frames = create_animation_frames_v2(figures)
        return {
            "success": True,
            "frames": frames,
            "total_frames": len(frames),
            "dist_x": dist_x,
            "dist_y": dist_y,
            "rho": rho,
            "n_total": n_total,
            "sample_corr": float(np.corrcoef(x, y)[0, 1])
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")



@app.post("/api/chapter4/digital-features-animation")
async def digital_features_animation(request):
    """数字特征可视化动画 - 返回帧序列"""
    try:
        data = await request.json()
        dist_type = data.get("dist_type", "正态分布")
        n_samples = data.get("n_samples", 10000)

        # 生成数据
        if dist_type == "正态分布":
            mean = data.get("mean", 0.0)
            std = data.get("std", 1.0)
            data_arr = np.random.normal(mean, std, n_samples)
            theory = {"mean": mean, "var": std**2, "std": std, "skew": 0.0, "kurt": 0.0}
        elif dist_type == "均匀分布":
            min_val = data.get("min_val", 0.0)
            max_val = data.get("max_val", 1.0)
            data_arr = np.random.uniform(min_val, max_val, n_samples)
            theory = {"mean": (min_val+max_val)/2, "var": (max_val-min_val)**2/12,
                     "std": np.sqrt((max_val-min_val)**2/12), "skew": 0.0, "kurt": -1.2}
        elif dist_type == "泊松分布":
            lam = data.get("lam", 5.0)
            data_arr = np.random.poisson(lam, n_samples)
            theory = {"mean": lam, "var": lam, "std": np.sqrt(lam),
                     "skew": 1/np.sqrt(lam) if lam > 0 else 0, "kurt": 1/lam if lam > 0 else 0}
        elif dist_type == "指数分布":
            rate = data.get("rate", 1.0)
            data_arr = np.random.exponential(1/rate, n_samples)
            theory = {"mean": 1/rate, "var": 1/(rate**2), "std": 1/rate, "skew": 2.0, "kurt": 6.0}
        else:  # 二项分布
            n = data.get("binom_n", 10)
            p = data.get("binom_p", 0.5)
            data_arr = np.random.binomial(n, p, n_samples)
            theory = {"mean": n*p, "var": n*p*(1-p), "std": np.sqrt(n*p*(1-p)),
                     "skew": (1-2*p)/np.sqrt(n*p*(1-p)) if n*p*(1-p) > 0 else 0,
                     "kurt": (1-6*p*(1-p))/(n*p*(1-p)) if n*p*(1-p) > 0 else 0}

        figures = []

        # 步骤1：总体分布生成（10帧）
        for frame in range(1, 11):
            progress = frame / 10
            fig, ax = plt.subplots(figsize=(8, 4))
            n_show = int(len(data_arr) * progress)
            if n_show > 0:
                ax.hist(data_arr[:n_show], bins=30, density=True, alpha=0.6,
                       color="steelblue", edgecolor="white")
            ax.axvline(x=theory["mean"], color='red', linestyle='--', linewidth=2,
                      label=f"理论均值={theory['mean']:.2f}")
            ax.set_title(f"{dist_type} 分布正在生成... ({n_show}/{len(data_arr)})")
            ax.set_xlabel("样本值")
            ax.set_ylabel("密度")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            figures.append(fig)

        # 步骤2：数字特征收敛（20帧）
        n_frames = 20
        n_values = np.unique(np.linspace(max(10, n_samples//50), n_samples, n_frames).astype(int))
        if n_values[0] > 10:
            n_values = np.concatenate(([10], n_values))
        if len(n_values) < 2:
            n_values = np.array([min(10, n_samples), n_samples])

        for idx, current_n in enumerate(n_values):
            current_data = data_arr[:current_n]
            hist_mean = np.mean(current_data)
            hist_var = np.var(current_data, ddof=1) if current_n >= 2 else 0
            hist_std = np.std(current_data, ddof=1) if current_n >= 2 else 0

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # 左上：当前样本直方图
            ax_hist = axes[0, 0]
            ax_hist.hist(current_data, bins=30, density=True, alpha=0.6,
                        color="steelblue", edgecolor="white")
            ax_hist.axvline(x=hist_mean, color='red', linestyle='-', linewidth=2,
                           label=f"样本均值={hist_mean:.3f}")
            ax_hist.axvline(x=theory["mean"], color='black', linestyle='--', linewidth=1.5,
                           alpha=0.7, label=f"理论均值={theory['mean']:.3f}")
            ax_hist.set_title(f"当前样本直方图 (n={current_n})")
            ax_hist.set_xlabel("样本值")
            ax_hist.set_ylabel("密度")
            ax_hist.legend(fontsize=8)
            ax_hist.grid(axis="y", alpha=0.3)

            # 右上：均值收敛
            ax_mean = axes[0, 1]
            ns = [n_values[k] for k in range(idx+1)]
            means = [np.mean(data_arr[:nn]) for nn in ns]
            ax_mean.plot(ns, means, 'o-', color='steelblue', markersize=4, linewidth=1.5)
            ax_mean.axhline(y=theory["mean"], color='red', linestyle='--', linewidth=2,
                           label=f"理论值={theory['mean']:.3f}")
            ax_mean.set_title("均值收敛")
            ax_mean.set_xlabel("样本量")
            ax_mean.set_ylabel("样本均值")
            ax_mean.legend(fontsize=8)
            ax_mean.grid(alpha=0.3)

            # 左下：方差收敛
            ax_var = axes[1, 0]
            variances = [np.var(data_arr[:nn], ddof=1) if nn >= 2 else 0 for nn in ns]
            ax_var.plot(ns, variances, 'o-', color='green', markersize=4, linewidth=1.5)
            ax_var.axhline(y=theory["var"], color='red', linestyle='--', linewidth=2,
                          label=f"理论值={theory['var']:.3f}")
            ax_var.set_title("方差收敛")
            ax_var.set_xlabel("样本量")
            ax_var.set_ylabel("样本方差")
            ax_var.legend(fontsize=8)
            ax_var.grid(alpha=0.3)

            # 右下：标准差收敛
            ax_std = axes[1, 1]
            stds = [np.std(data_arr[:nn], ddof=1) if nn >= 2 else 0 for nn in ns]
            ax_std.plot(ns, stds, 'o-', color='orange', markersize=4, linewidth=1.5)
            ax_std.axhline(y=theory["std"], color='red', linestyle='--', linewidth=2,
                          label=f"理论值={theory['std']:.3f}")
            ax_std.set_title("标准差收敛")
            ax_std.set_xlabel("样本量")
            ax_std.set_ylabel("样本标准差")
            ax_std.legend(fontsize=8)
            ax_std.grid(alpha=0.3)

            fig.suptitle(f"数字特征正在收敛... ({idx+1}/{len(n_values)})", fontsize=14)
            fig.tight_layout()
            figures.append(fig)

        frames = create_animation_frames_v2(figures)
        return {
            "success": True,
            "frames": frames,
            "total_frames": len(frames),
            "dist_type": dist_type,
            "n_samples": n_samples,
            "theory_mean": float(theory["mean"]),
            "theory_var": float(theory["var"]),
            "theory_std": float(theory["std"]),
            "theory_skew": float(theory["skew"]),
            "theory_kurt": float(theory["kurt"])
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")



@app.post("/api/chapter6/order-stats-animation")
async def order_stats_animation(request):
    """次序统计量可视化动画 - 返回帧序列"""
    try:
        data = await request.json()
        dist_type = data.get("dist_type", "正态分布")
        sample_size = data.get("sample_size", 15)

        np.random.seed(42)
        if dist_type == "正态分布":
            data_arr = np.random.normal(0, 1, sample_size)
        elif dist_type == "均匀分布":
            data_arr = np.random.uniform(0, 1, sample_size)
        else:  # 指数分布
            data_arr = np.random.exponential(1, sample_size)

        sorted_data = np.sort(data_arr)
        figures = []

        # 步骤1：原始样本（10帧）
        for frame in range(1, 11):
            progress = frame / 10
            n_show = int(sample_size * progress)
            fig, ax = plt.subplots(figsize=(12, 4))
            if n_show > 0:
                y_jitter = np.random.uniform(-0.15, 0.15, n_show)
                ax.scatter(data_arr[:n_show], y_jitter, s=120, c='steelblue',
                          edgecolors='white', linewidth=1.5, alpha=0.9, zorder=5)
                for j in range(n_show):
                    ax.text(data_arr[j], y_jitter[j] + 0.22, f'{data_arr[j]:.2f}',
                           ha='center', va='bottom', fontsize=8, alpha=0.7)
            ax.set_title(f"正在生成样本... ({n_show}/{sample_size})  {dist_type}")
            ax.set_xlabel("样本值")
            ax.set_yticks([])
            ax.set_ylim(-0.6, 0.6)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            figures.append(fig)

        # 步骤2：排序动画（20帧）
        n_frames = min(sample_size, 20)
        indices = np.linspace(1, sample_size, n_frames).astype(int)
        if indices[0] == 0:
            indices[0] = 1
        for idx, k in enumerate(indices):
            fig, ax = plt.subplots(figsize=(12, 6))
            y_jitter = np.random.uniform(0.3, 0.6, sample_size)
            ax.scatter(data_arr, y_jitter, s=80, c='lightgray',
                      edgecolors='white', linewidth=1, alpha=0.4, zorder=1)
            sorted_subset = sorted_data[:k]
            y_sorted = np.full(k, -0.3)
            colors = plt.cm.viridis(np.linspace(0, 1, k))
            ax.scatter(sorted_subset, y_sorted, s=150, c=colors,
                      edgecolors='black', linewidth=1.5, alpha=0.95, zorder=5)
            for j in range(k):
                ax.text(sorted_subset[j], -0.55, f'$X_{{({j+1})}}$',
                       ha='center', va='top', fontsize=9, color='darkblue')
                ax.text(sorted_subset[j], -0.05, f'{sorted_subset[j]:.2f}',
                       ha='center', va='bottom', fontsize=8, alpha=0.8)
            ax.set_title(f"正在排队... ({k}/{sample_size}) 已排序 {k} 个")
            ax.set_xlabel("样本值")
            ax.set_yticks([])
            ax.set_ylim(-0.8, 0.8)
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            figures.append(fig)

        # 步骤3：标记关键次序统计量（10帧）
        for frame in range(1, 11):
            alpha = frame / 10
            fig, ax = plt.subplots(figsize=(12, 5))
            y_sorted = np.zeros(sample_size)
            ax.scatter(sorted_data, y_sorted, s=150, c='steelblue',
                      edgecolors='white', linewidth=1.5, alpha=0.6, zorder=3)

            mid_idx = sample_size // 2 if sample_size % 2 == 1 else sample_size // 2 - 1
            if frame >= 4:
                ax.scatter(sorted_data[0], 0, s=300, c='red', marker='*',
                          edgecolors='darkred', linewidth=2, alpha=min(1, (frame-3)/3),
                          zorder=10, label=f'最小值 $X_{{(1)}}$ = {sorted_data[0]:.3f}')
            if frame >= 7:
                ax.scatter(sorted_data[mid_idx], 0, s=300, c='orange', marker='*',
                          edgecolors='darkorange', linewidth=2, alpha=min(1, (frame-6)/3),
                          zorder=10, label=f'中位数 = {sorted_data[mid_idx]:.3f}')
            if frame >= 10:
                ax.scatter(sorted_data[-1], 0, s=300, c='purple', marker='*',
                          edgecolors='indigo', linewidth=2, alpha=1,
                          zorder=10, label=f'最大值 $X_{{({sample_size})}}$ = {sorted_data[-1]:.3f}')

            ax.set_title(f"关键次序统计量正在高亮... ({frame}/10)")
            ax.set_xlabel("样本值")
            ax.set_yticks([])
            ax.set_ylim(-0.5, 0.5)
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(axis='x', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            figures.append(fig)

        frames = create_animation_frames_v2(figures)
        return {
            "success": True,
            "frames": frames,
            "total_frames": len(frames),
            "dist_type": dist_type,
            "sample_size": sample_size,
            "min": float(sorted_data[0]),
            "median": float(sorted_data[mid_idx]),
            "max": float(sorted_data[-1]),
            "range": float(sorted_data[-1] - sorted_data[0])
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")



@app.post("/api/chapter7/moment-est-animation")
async def moment_est_animation(request):
    """矩估计可视化动画 - 返回帧序列"""
    try:
        data = await request.json()
        a_true = data.get("a_true", 0.0)
        b_true = data.get("b_true", 1.0)
        n = data.get("n", 100)

        np.random.seed(42)
        data_arr = np.random.uniform(a_true, b_true, n)
        history = []
        for i in range(1, n + 1):
            subset = data_arr[:i]
            mu1_i = np.mean(subset)
            mu2_i = np.mean(subset ** 2)
            var_i = mu2_i - mu1_i ** 2
            if var_i < 0 and var_i > -1e-12:
                var_i = 0.0
            if var_i >= 0:
                a_i = mu1_i - np.sqrt(3 * var_i)
                b_i = mu1_i + np.sqrt(3 * var_i)
            else:
                a_i, b_i = np.nan, np.nan
            history.append({'a': a_i, 'b': b_i, 'mu1': mu1_i})

        figures = []

        # 步骤1：真实总体（10帧）
        for frame in range(1, 11):
            progress = frame / 10
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.linspace(a_true - 0.2, b_true + 0.2, 200)
            y = np.where((x >= a_true) & (x <= b_true), 1/(b_true - a_true), 0)
            ax.plot(x, y, 'steelblue', linewidth=2)
            ax.fill_between(x, y, alpha=0.3 * progress, color='steelblue')
            ax.axvline(a_true, color='black', linestyle='-', linewidth=2, label=f'真实 a={a_true}')
            ax.axvline(b_true, color='black', linestyle='-', linewidth=2, label=f'真实 b={b_true}')
            ax.set_title(f"真实总体 U({a_true}, {b_true}) 正在显现... ({frame}/10)")
            ax.set_xlabel("x")
            ax.set_ylabel("概率密度")
            ax.legend()
            ax.grid(alpha=0.3)
            figures.append(fig)

        # 步骤2：逐个加入样本，估计值跳动收敛（20帧）
        frames_list = list(range(1, 21)) + list(np.linspace(21, n, 20).astype(int))
        frames_list = sorted(set(frames_list))
        for idx, i in enumerate(frames_list):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})
            ax_main = axes[0]
            ax_conv = axes[1]

            subset = data_arr[:i]
            ax_main.hist(subset, bins=min(30, max(10, i//3)), alpha=0.6,
                        color="steelblue", edgecolor="white", density=True)
            h = history[i - 1]
            if not (np.isnan(h['a']) or np.isnan(h['b'])):
                ax_main.axvline(h['a'], color='red', linestyle='--', linewidth=2.5,
                               label=f"â={h['a']:.3f}")
                ax_main.axvline(h['b'], color='green', linestyle='--', linewidth=2.5,
                               label=f"b̂={h['b']:.3f}")
                if h['b'] > h['a']:
                    x_est = np.linspace(h['a'], h['b'], 100)
                    y_est = np.full_like(x_est, 1/(h['b'] - h['a']))
                    ax_main.plot(x_est, y_est, 'r-', alpha=0.4, linewidth=2)
            ax_main.axvline(a_true, color='black', linestyle='-', alpha=0.3, label=f'真实 a={a_true}')
            ax_main.axvline(b_true, color='black', linestyle='-', alpha=0.3, label=f'真实 b={b_true}')
            ax_main.set_title(f"当前样本 n={i} —— 矩估计区间")
            ax_main.set_xlabel("样本值")
            ax_main.set_ylabel("密度")
            ax_main.legend(fontsize=8)
            ax_main.grid(axis="y", alpha=0.3)

            a_traj = [hh['a'] for hh in history[:i]]
            b_traj = [hh['b'] for hh in history[:i]]
            ax_conv.plot(range(1, i + 1), a_traj, 'r.-', markersize=4, linewidth=1.2, label='â (估计)')
            ax_conv.plot(range(1, i + 1), b_traj, 'g.-', markersize=4, linewidth=1.2, label='b̂ (估计)')
            ax_conv.axhline(a_true, color='black', linestyle='-', alpha=0.4, label=f'真实 a={a_true}')
            ax_conv.axhline(b_true, color='black', linestyle='-', alpha=0.4, label=f'真实 b={b_true}')
            ax_conv.set_title("估计值收敛轨迹")
            ax_conv.set_xlabel("样本量")
            ax_conv.set_ylabel("参数估计值")
            ax_conv.legend(fontsize=8)
            ax_conv.grid(alpha=0.3)

            fig.suptitle(f"矩估计收敛中... ({idx + 1}/{len(frames_list)})", fontsize=13)
            fig.tight_layout()
            figures.append(fig)

        frames = create_animation_frames_v2(figures)
        h_final = history[-1]
        return {
            "success": True,
            "frames": frames,
            "total_frames": len(frames),
            "a_true": a_true,
            "b_true": b_true,
            "n": n,
            "a_hat": float(h_final['a']) if not np.isnan(h_final['a']) else None,
            "b_hat": float(h_final['b']) if not np.isnan(h_final['b']) else None,
            "error": float(abs(h_final['a'] - a_true) + abs(h_final['b'] - b_true)) if not np.isnan(h_final['a']) else None
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")



@app.post("/api/chapter7/mle-animation")
async def mle_animation(request):
    """最大似然估计可视化动画 - 返回帧序列"""
    try:
        data = await request.json()
        mu_true = data.get("mu_true", 5.0)
        sigma_true = data.get("sigma_true", 2.0)
        n = data.get("n", 100)

        np.random.seed(42)
        data_arr = np.random.normal(mu_true, sigma_true, n)
        history = []
        for i in range(1, n + 1):
            subset = data_arr[:i]
            mu_i = np.mean(subset)
            var_i = np.mean((subset - mu_i) ** 2)
            sigma_i = np.sqrt(var_i) if var_i > 0 else 0.0
            history.append({'mu': mu_i, 'sigma': sigma_i})

        figures = []

        # 步骤1：真实总体（10帧）
        for frame in range(1, 11):
            progress = frame / 10
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.linspace(mu_true - 4 * sigma_true, mu_true + 4 * sigma_true, 200)
            y = stats.norm.pdf(x, mu_true, sigma_true)
            ax.plot(x, y, 'steelblue', linewidth=2)
            ax.fill_between(x, y, alpha=0.3 * progress, color='steelblue')
            ax.axvline(mu_true, color='black', linestyle='-', linewidth=2, label=f'真实 μ={mu_true}')
            ax.set_title(f"真实总体 N({mu_true}, {sigma_true}²) 正在显现... ({frame}/10)")
            ax.set_xlabel("x")
            ax.set_ylabel("概率密度")
            ax.legend()
            ax.grid(alpha=0.3)
            figures.append(fig)

        # 步骤2：逐个加入样本，MLE跳动收敛（20帧）
        frames_list = list(range(1, 21)) + list(np.linspace(21, n, 20).astype(int))
        frames_list = sorted(set(frames_list))
        for idx, i in enumerate(frames_list):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})
            ax_main = axes[0]
            ax_conv = axes[1]

            subset = data_arr[:i]
            ax_main.hist(subset, bins=min(30, max(10, i // 3)), alpha=0.6,
                        color="steelblue", edgecolor="white", density=True)
            h = history[i - 1]
            if h['sigma'] > 0:
                x_est = np.linspace(h['mu'] - 3 * h['sigma'], h['mu'] + 3 * h['sigma'], 200)
                y_est = stats.norm.pdf(x_est, h['mu'], h['sigma'])
                ax_main.plot(x_est, y_est, 'r-', linewidth=2.5, alpha=0.8,
                            label=f"MLE: N({h['mu']:.2f}, {h['sigma']:.2f}²)")
            x_true_range = np.linspace(mu_true - 3 * sigma_true, mu_true + 3 * sigma_true, 200)
            y_true = stats.norm.pdf(x_true_range, mu_true, sigma_true)
            ax_main.plot(x_true_range, y_true, 'g--', linewidth=2, alpha=0.6,
                        label=f"真实: N({mu_true}, {sigma_true}²)")
            ax_main.axvline(mu_true, color='black', linestyle='-', alpha=0.3)
            ax_main.set_title(f"当前样本 n={i} —— MLE 拟合曲线")
            ax_main.set_xlabel("样本值")
            ax_main.set_ylabel("密度")
            ax_main.legend(fontsize=8)
            ax_main.grid(axis="y", alpha=0.3)

            mu_traj = [hh['mu'] for hh in history[:i]]
            sigma_traj = [hh['sigma'] for hh in history[:i]]
            ax_conv.plot(range(1, i + 1), mu_traj, 'b.-', markersize=4, linewidth=1.2, label='μ̂ (均值)')
            ax_conv.plot(range(1, i + 1), sigma_traj, 'r.-', markersize=4, linewidth=1.2, label='σ̂ (标准差)')
            ax_conv.axhline(mu_true, color='blue', linestyle='--', alpha=0.4, label=f'真实 μ={mu_true}')
            ax_conv.axhline(sigma_true, color='red', linestyle='--', alpha=0.4, label=f'真实 σ={sigma_true}')
            ax_conv.set_title("MLE 参数收敛轨迹")
            ax_conv.set_xlabel("样本量")
            ax_conv.set_ylabel("参数估计值")
            ax_conv.legend(fontsize=8)
            ax_conv.grid(alpha=0.3)

            fig.suptitle(f"MLE 收敛中... ({idx + 1}/{len(frames_list)})", fontsize=13)
            fig.tight_layout()
            figures.append(fig)

        frames = create_animation_frames_v2(figures)
        h_final = history[-1]
        return {
            "success": True,
            "frames": frames,
            "total_frames": len(frames),
            "mu_true": mu_true,
            "sigma_true": sigma_true,
            "n": n,
            "mu_hat": float(h_final['mu']),
            "sigma_hat": float(h_final['sigma']),
            "error": float(abs(h_final['mu'] - mu_true) + abs(h_final['sigma'] - sigma_true))
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")



@app.post("/api/chapter8/ci-animation")
async def ci_animation(request):
    """置信区间可视化动画 - 返回帧序列"""
    try:
        data = await request.json()
        sample_size = data.get("sample_size", 50)
        confidence_level = data.get("confidence_level", 0.95)

        population_mean = 50.0
        population_std = 10.0
        n_repeats = 100

        all_cis = []
        for _ in range(n_repeats):
            sample = np.random.normal(population_mean, population_std, sample_size)
            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)
            alpha = 1 - confidence_level
            t_value = stats.t.ppf(1 - alpha / 2, df=sample_size - 1)
            margin_error = t_value * sample_std / np.sqrt(sample_size)
            ci_lower = sample_mean - margin_error
            ci_upper = sample_mean + margin_error
            contains = ci_lower <= population_mean <= ci_upper
            all_cis.append({
                'mean': sample_mean,
                'lower': ci_lower,
                'upper': ci_upper,
                'contains': contains
            })

        figures = []
        first = all_cis[0]

        # 步骤1：单次实验演示（10帧）
        for frame in range(1, 11):
            progress = frame / 10
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axvline(x=population_mean, color='black', linestyle='-.', linewidth=2,
                      label=f'总体均值 μ={population_mean}', zorder=5)
            if progress > 0.3:
                ax.axvline(x=first['mean'], color='red', linestyle='-', linewidth=2,
                          alpha=min(1, (progress - 0.3) / 0.3),
                          label=f'样本均值 x̄={first["mean"]:.2f}', zorder=4)
            if progress > 0.6:
                alpha_ci = min(1, (progress - 0.6) / 0.3)
                color_ci = '#2ecc71' if first['contains'] else '#e74c3c'
                ax.axvspan(first['lower'], first['upper'], alpha=alpha_ci * 0.3,
                          color=color_ci, zorder=1)
                ax.axvline(x=first['lower'], color=color_ci, linestyle='--', linewidth=2,
                          alpha=alpha_ci, zorder=3)
                ax.axvline(x=first['upper'], color=color_ci, linestyle='--', linewidth=2,
                          alpha=alpha_ci, zorder=3)
            ax.set_xlim(population_mean - 15, population_mean + 15)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_title(f"构建置信区间中... ({frame}/10)")
            ax.set_xlabel("样本值")
            ax.legend(loc='upper right')
            ax.grid(alpha=0.3)
            figures.append(fig)

        # 步骤2：100次重复实验，CI线段纵向排列（20帧）
        for frame in range(1, 21):
            n_show = int(len(all_cis) * frame / 20)
            current_cis = all_cis[:n_show]

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.axvline(x=population_mean, color='black', linestyle='-', linewidth=2,
                      label=f'总体均值 μ={population_mean}', zorder=5)

            for i, ci in enumerate(current_cis):
                color = '#2ecc71' if ci['contains'] else '#e74c3c'
                ax.plot([ci['lower'], ci['upper']], [i, i], color=color,
                       linewidth=2.5, solid_capstyle='round', zorder=3)
                ax.plot(ci['mean'], i, 'o', color='white', markersize=4,
                       markeredgecolor='black', markeredgewidth=0.8, zorder=4)

            n_contains = sum(1 for c in current_cis if c['contains'])
            ax.set_title(f"置信区间正在生成... ({n_show}/{len(all_cis)})  " +
                        f"命中率: {n_contains}/{n_show} = {n_contains/n_show*100:.1f}%" if n_show > 0 else "")
            ax.set_xlabel("样本值")
            ax.set_ylabel("实验序号")
            ax.set_xlim(population_mean - 15, population_mean + 15)
            ax.set_ylim(-2, len(all_cis) + 2)
            ax.invert_yaxis()

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', edgecolor='none', label='包含真实均值'),
                Patch(facecolor='#e74c3c', edgecolor='none', label='不包含真实均值')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
            ax.grid(alpha=0.3)
            figures.append(fig)

        frames = create_animation_frames_v2(figures)
        n_contains = sum(1 for c in all_cis if c['contains'])
        coverage = n_contains / len(all_cis) * 100
        return {
            "success": True,
            "frames": frames,
            "total_frames": len(frames),
            "sample_size": sample_size,
            "confidence_level": confidence_level,
            "population_mean": population_mean,
            "coverage": float(coverage),
            "n_contains": n_contains,
            "n_total": len(all_cis)
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成动画失败: {str(e)}")

