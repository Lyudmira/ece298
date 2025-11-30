
# ECE-298 Lab B5 Operating Plan: 任务清单与分析框架 (README)

## 1. 项目核心目标 (Project Objective)

本项目的核心目标是为梯田灌溉系统的原型机制定一份 **24小时运营计划 (Operating Plan)**，并将其填入指定的 Excel 电子表格中。

**成功的定义 (基于 Rubric):**
1.  **完整性**：必须覆盖完整的 24 小时周期（不可有空白，不可超时）。
2.  **准确性**：必须完成所有区域的需水量（Inlet + 3 Zones）。
    *   *硬性约束*：每个区域的供水误差不得超过 **100 加仑**。
3.  **无浪费**：不得泵送超过需求的水量（即浪费水资源）。
4.  **经济最优**：在满足上述物理约束的前提下，使 **24小时总运营成本 ($)** 最低。

---

## 2. 已知基础数据 (Baseline Data)

这些数据来源于项目说明文档 (PDF Pages 1, 4, 11)，是不可更改的常量。

### A. 需水量 (Water Requirements)
*   **Inlet (蓄水池补水)**: `94,000` Gallons (扬程 40ft)
*   **Zone 3**: `14,000` Gallons (扬程 60ft)
*   **Zone 2**: `39,000` Gallons (扬程 50ft)
*   **Zone 1**: `41,000` Gallons (扬程 25ft)
*   **总任务量**: `188,000` Gallons

### B. 机械参数 (Mechanical Specs)
*   **齿轮箱比例**: `2.5 : 1`
    *   公式: `Motor RPM = 2.5 × Pump RPM`
    *   *注意：所有功率图表基于 Motor RPM，所有流体力学图表基于 Pump RPM。*

### C. 经济参数 (Time of Use Pricing)
*   **Off-Peak (低谷)**: `7.4` ¢/kWh —— **19:00 至 07:00** (12小时)
*   **Mid-Peak (平峰)**: `10.2` ¢/kWh —— **07:00-11:00** & **17:00-19:00**
*   **On-Peak (高峰)**: `15.1` ¢/kWh —— **11:00 至 17:00** (6小时)

---

## 3. 核心算式模型 (Mathematical Model)

为了找到最优解，我们需要构建以下变量之间的计算链条：

### 变量定义
*   $R_p$: 泵转速 (Pump RPM) —— **这是我们唯一的自变量 (我们控制它)**
*   $R_m$: 电机转速 (Motor RPM)
*   $Q$: 流量 (Flow Rate in GPM)
*   $P$: 功率 (Power in kW)
*   $T$: 运行时间 (Time in Hours)
*   $V$: 目标水量 (Volume in Gallons)
*   $E$: 能耗 (Energy in kWh)
*   $C$: 成本 (Cost in $)

### 计算逻辑链
1.  **转速转换**:
    $$R_m = 2.5 \times R_p$$
2.  **流量获取 (查表/插值)**:
    $$Q = f_{system\_curve}(R_p)$$
    *(来源：泵曲线与系统曲线的交点)*
3.  **功率获取 (查表/插值)**:
    $$P = f_{power\_curve}(R_m)$$
    *(来源：电机功率曲线)*
4.  **时间计算**:
    $$T = \frac{V}{Q \times 60}$$
    *(注意：GPM 是每分钟，时间单位是小时，需除以 60)*
5.  **能耗计算**:
    $$E = P \times T$$
6.  **成本计算 (积分)**:
    $$C = \int_{t_{start}}^{t_{end}} P \times Price(t) \,dt$$
    *(即：功率 × 时间 × 对应时段的电价)*

### 优化目标函数
$$ \text{Minimize } C_{total} = C_{inlet} + C_{zone1} + C_{zone2} + C_{zone3} $$
**约束条件**:
1.  $\sum T \le 24$ (总时间不超过24小时)
2.  $Q(R_p) > 0$ (必须能推动水流，即扬程足够)

---

## 4. 待解决问题与计算任务 (Analysis Tasks)

为了达成最优，我们不能盲目填数。我们需要按步骤执行以下计算任务：

### 任务 1：建立高精度查找表 (Lookup Table Construction)
*   **现状**：PDF 图表是粗略的，但我们已经通过获得了关键点（见文末，table 1）
*   **行动**：
    1.  利用 Python 对采样数据进行拟合（Curve Fitting），不需要完全精确，不需要一个曲线对应一个函数，插值算法是可以接受的，如果算法能理解曲线的变化略微精确一点更好，但不强求。（注意inlet curve在GPM > 160 之后表现出了反常大幅增加而不符合之前的曲线（温和的增长接近直线）；这也是我不建议你一个曲线拟合一个函数的原因。
 
       
    3.  生成两个核心函数的映射关系：
        *   `get_flow(pump_rpm, zone_id) -> GPM`
        *   `get_power(pump_rpm) -> kW`
    4.  **目的**：让我们能输入任意一个 RPM（比如 87 RPM），立即得到精确的 GPM 和 kW，而不是靠猜。

### 任务 2：计算“能效甜点” (Efficiency Sweet Spot Analysis)
*   **定义**：能效指标 $\eta = \text{Gallons per kWh}$。
    $$ \eta = \frac{Q \times 60}{P} $$
*   **行动**：
    1.  对于每个区域 (Zone 1, 2, 3, Inlet)，遍历所有可能的 RPM (40-100)。
    2.  画出 `RPM vs Efficiency` 曲线。
    3.  找到每个区域的 $\eta_{max}$ 对应的 RPM。
*   **预期发现**：
    *   是否存在一个低转速区间，虽然慢，但极其省电？
    *   Zone 3 是否存在一个“截止频率”，低于此转速效率急剧归零？

### 任务 3：时间-成本博弈 (Time-Cost Trade-off)
*   **核心矛盾**：
    *   用最省电的 RPM（通常较低）运行 $\rightarrow$ 总耗时可能超过 30 小时 $\rightarrow$ 任务失败。
    *   全速运行 $\rightarrow$ 耗时短，但功率呈指数爆炸 $\rightarrow$ 成本过高。
*   **计算行动**：
    1.  计算“最低可行速度”方案的总耗时。如果 > 24h，则必须提速。
    2.  计算“Off-Peak 优先”方案：为了把 Inlet (94,000 gal) 塞进 12小时的低谷期，最低需要多少 RPM？
    3.  计算“On-Peak 止损”方案：如果必须在白天 (高价) 运行，是用高转速快点结束划算，还是用低转速慢慢磨划算？（比较 $P \times T \times Price$）。

### 任务 4：生成最终排班表 (Scheduling)
*   **行动**：
    1.  确定四个任务的执行顺序（通常大任务优先抢占 Off-Peak）。
    2.  确定每个任务的具体 RPM。
    3.  精确计算 Start Time 和 End Time。
    4.  验证供水量误差是否在 100 加仑以内（微调分钟数）。

---

## 5. 总结：我们的“最优”到底是什么？

在这个项目中，“最优”不是一个单一的数学极值，而是一个**受限条件下的策略选择**。

**最优设计必须满足：**
1.  **物理可行**：水能推上去。
2.  **时间合规**：总耗时 必须小于24小时。
3.  **无浪费**：不得泵送超过需求的水量（即浪费水资源）。
4.  **经济最优**：在满足上述物理约束的前提下，使 **24小时总运营成本 ($)** 最低。




Table 1:
   
注意，这个图表标注的终点就是真正的终点，定义域的尽头。

20, 40, 60, 80, 100rpm 的定义域开始于 0 flow (GPM)，不过此刻的head tdh没有被测量——需要的时候你可以自己估计）

Flow (GPM), Head TDH (ft)
---

## 100rpm（sorted by gpm）

0.34, 81.87
38.51, 77.63
71.98, 72.87
96.08, 68.01
120.68, 62.51
139.28, 57.31
163.88, 49.03
170.15, 46.44
184.99, 39.77
193.07, 35.34
202.47, 30.26

## 80rpm（sorted by gpm）

0.13, 67.60
29.34, 64.59
64.09, 59.43
105.81, 50.87
110.15, 49.44
120.40, 46.79
143.20, 39.48
151.83, 36.45
171.38, 27.62
182.43, 21.65

## 60rpm（sorted by gpm）

0.13, 51.47
28.49, 48.46
57.70, 44.25
86.06, 38.44
98.54, 35.37
123.56, 27.61
143.84, 20.23
155.57, 14.78

## 40rpm（sorted by gpm）

0.55, 32.93
45.12, 27.75
58.36, 25.30
68.71, 23.60
87.97, 19.23
110.79, 13.30
128.91, 7.39

## 20rpm（sorted by gpm）

0.55, 19.46
17.61, 17.95
37.52, 16.11
43.91, 14.85
55.45, 12.32
71.56, 7.14
79.45, 4.59

---

## blue (3)（sorted by gpm）

0.00, 0.00
9.08, 3.98
18.68, 8.08
30.41, 13.12
37.52, 16.11
45.33, 19.35
58.36, 25.30
71.34, 31.01
86.06, 38.44
97.78, 45.34
105.81, 50.87
120.68, 62.51
128.70, 68.86
134.24, 73.38

## green (2)（sorted by gpm）

0.00, 0.00
12.71, 4.63
17.40, 6.09
25.29, 8.73
36.16, 12.30
43.91, 14.85
53.01, 18.13
68.71, 23.60
79.23, 27.26
90.53, 31.76
98.54, 35.37
107.59, 39.98
120.40, 46.79
133.18, 53.91
139.28, 57.31
143.41, 59.75

## purple (INLET)（sorted by gpm）

0.00, 0.00
5.25, 1.45
10.15, 2.51
22.09, 5.27
42.56, 9.49
55.45, 12.32
66.23, 14.76
80.51, 17.65
87.97, 19.23
105.67, 23.18
116.12, 25.82
123.56, 27.61
138.29, 31.50
151.83, 36.45
163.67, 42.23
170.15, 46.44
180.09, 56.05

## red (1)（sorted by gpm）

0.00, 0.00
13.13, 0.90
21.45, 1.54
28.91, 2.46
40.85, 3.49
51.73, 4.80
60.04, 5.71
71.56, 7.14
79.87, 8.45
90.96, 10.02
110.79, 13.30
124.01, 15.94
137.01, 18.57
143.84, 20.23
151.09, 22.00
164.31, 25.43
171.38, 27.62
181.58, 30.98
193.07, 35.34
202.47, 39.46
208.87, 42.64

---

Table 2:

y: MOTOR POWER (KW)
x: MOTOR RPM

$\hat{y} \approx 7.2803 \times 10^{-6}(x+13.092)^{3.18385}$ ( 0 <= x < 250.5 )



---

---



可以参考的

Thinking
Implementing pump flow and power calculations
import numpy as np, math, pandas as pd

Loading pump and system curve data
pump_data = {
100: [(0.34,81.87),(38.51,77.63),(71.98,72.87),(96.08,68.01),(120.68,62.51),(139.28,57.31),(163.88,49.03),(170.15,46.44),(184.99,39.77),(193.07,35.34),(202.47,30.26)],
80: [(0.13,67.60),(29.34,64.59),(64.09,59.43),(105.81,50.87),(110.15,49.44),(120.40,46.79),(143.20,39.48),(151.83,36.45),(171.38,27.62),(182.43,21.65)],
60: [(0.13,51.47),(28.49,48.46),(57.70,44.25),(86.06,38.44),(98.54,35.37),(123.56,27.61),(143.84,20.23),(155.57,14.78)],
40: [(0.55,32.93),(45.12,27.75),(58.36,25.30),(68.71,23.60),(87.97,19.23),(110.79,13.30),(128.91,7.39)],
20: [(0.55,19.46),(17.61,17.95),(37.52,16.11),(43.91,14.85),(55.45,12.32),(71.56,7.14),(79.45,4.59)]
}
sys_data = {
"zone3":[(0.00,0.00),(9.08,3.98),(18.68,8.08),(30.41,13.12),(37.52,16.11),(45.33,19.35),(58.36,25.30),(71.34,31.01),(86.06,38.44),(97.78,45.34),(105.81,50.87),(120.68,62.51),(128.70,68.86),(134.24,73.38)],
"zone2":[(0.00,0.00),(12.71,4.63),(17.40,6.09),(25.29,8.73),(36.16,12.30),(43.91,14.85),(53.01,18.13),(68.71,23.60),(79.23,27.26),(90.53,31.76),(98.54,35.37),(107.59,39.98),(120.40,46.79),(133.18,53.91),(139.28,57.31),(143.41,59.75)],
"inlet":[(0.00,0.00),(5.25,1.45),(10.15,2.51),(22.09,5.27),(42.56,9.49),(55.45,12.32),(66.23,14.76),(80.51,17.65),(87.97,19.23),(105.67,23.18),(116.12,25.82),(123.56,27.61),(138.29,31.50),(151.83,36.45),(163.67,42.23),(170.15,46.44),(180.09,56.05)],
"zone1":[(0.00,0.00),(13.13,0.90),(21.45,1.54),(28.91,2.46),(40.85,3.49),(51.73,4.80),(60.04,5.71),(71.56,7.14),(79.87,8.45),(90.96,10.02),(110.79,13.30),(124.01,15.94),(137.01,18.57),(143.84,20.23),(151.09,22.00),(164.31,25.43),(171.38,27.62),(181.58,30.98),(193.07,35.34),(202.47,39.46),(208.87,42.64)]
}
pump_data, sys_data.keys()

({100: [(0.34, 81.87),
   (38.51, 77.63),
   (71.98, 72.87),
   (96.08, 68.01),
   (120.68, 62.51),
   (139.28, 57.31),
   (163.88, 49.03),
   (170.15, 46.44),
   (184.99, 39.77),
   (193.07, 35.34),
   (202.47, 30.26)],
  80: [(0.13, 67.6),
   (29.34, 64.59),
   (64.09, 59.43),
   (105.81, 50.87),
   (110.15, 49.44),
   (120.4, 46.79),
   (143.2, 39.48),
   (151.83, 36.45),
   (171.38, 27.62),
   (182.43, 21.65)],
  60: [(0.13, 51.47),
   (28.49, 48.46),
   (57.7, 44.25),
   (86.06, 38.44),
   (98.54, 35.37),
   (123.56, 27.61),
   (143.84, 20.23),
   (155.57, 14.78)],
  40: [(0.55, 32.93),
   (45.12, 27.75),
   (58.36, 25.3),
   (68.71, 23.6),
   (87.97, 19.23),
   (110.79, 13.3),
   (128.91, 7.39)],
  20: [(0.55, 19.46),
   (17.61, 17.95),
   (37.52, 16.11),
   (43.91, 14.85),
   (55.45, 12.32),
   (71.56, 7.14),
   (79.45, 4.59)]},
 dict_keys(['zone3', 'zone2', 'inlet', 'zone1']))
Implementing monotone cubic Hermite interpolation
for r in [20,40,60,80,100]:
    H0 = pump_data[r][0][1]
    print(r, H0, H0/(r**2))

20 19.46 0.04865
40 32.93 0.02058125
60 51.47 0.014297222222222222
80 67.6 0.010562499999999999
100 81.87 0.008187
Computing pump head interpolation and intersection with system curve
def pchip_slopes(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)
    # interior points
    for k in range(1, n-1):
        if delta[k-1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k-1]) != np.sign(delta[k]):
            d[k] = 0.0
        else:
            w1 = 2*h[k] + h[k-1]
            w2 = h[k] + 2*h[k-1]
            d[k] = (w1 + w2) / (w1/delta[k-1] + w2/delta[k])
    # endpoints
    # d0
    d0 = ((2*h[0] + h[1])*delta[0] - h[0]*delta[1])/(h[0]+h[1]) if n>2 else delta[0]
    if np.sign(d0) != np.sign(delta[0]):
        d0 = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(d0) > abs(3*delta[0]):
        d0 = 3*delta[0]
    d[0]=d0
    # dn
    dn = ((2*h[-1] + h[-2])*delta[-1] - h[-1]*delta[-2])/(h[-1]+h[-2]) if n>2 else delta[-1]
    if np.sign(dn) != np.sign(delta[-1]):
        dn = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(dn) > abs(3*delta[-1]):
        dn = 3*delta[-1]
    d[-1]=dn
    return d

def pchip_interp(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]; y = y[order]
    # remove duplicates by averaging
    ux, inv = np.unique(x, return_inverse=True)
    if len(ux) < len(x):
        y_acc = np.zeros_like(ux, dtype=float)
        counts = np.zeros_like(ux, dtype=int)
        for i, idx in enumerate(inv):
            y_acc[idx] += y[i]; counts[idx]+=1
        y = y_acc / counts
        x = ux
    d = pchip_slopes(x,y)
    h = np.diff(x)
    def f(xq):
        xq = np.asarray(xq, dtype=float)
        # allocate
        yq = np.empty_like(xq, dtype=float)
        # for each xq, find interval
        idx = np.searchsorted(x, xq) - 1
        idx = np.clip(idx, 0, len(x)-2)
        x0=x[idx]; x1=x[idx+1]
        y0=y[idx]; y1=y[idx+1]
        d0=d[idx]; d1=d[idx+1]
        h01 = x1-x0
        t = (xq-x0)/h01
        # hermite basis
        h00 = (2*t**3 - 3*t**2 + 1)
        h10 = (t**3 - 2*t**2 + t)
        h01b= (-2*t**3 + 3*t**2)
        h11 = (t**3 - t**2)
        yq = h00*y0 + h10*h01*d0 + h01b*y1 + h11*h01*d1
        return yq
    return f, (x,y,d)

# test quickly on monotonic data
x=[0,1,2,3]; y=[0,1,1.5,2]
f,_=pchip_interp(x,y)
f(np.array([0,0.5,1,1.5,2.5,3]))

array([0.        , 0.57291667, 1.        , 1.27083333, 1.75      ,
       2.        ])
Building pump curve representation in normalized s
# build pchip for each rpm in normalized s = Q/Qmax
pump_curves = {}
pump_qmax = {}
for r, pts in pump_data.items():
    q=np.array([p[0] for p in pts])
    h=np.array([p[1] for p in pts])
    qmax=q.max()
    pump_qmax[r]=qmax
    s=q/qmax
    f, meta = pchip_interp(s, h)  # head vs s
    pump_curves[r]=f

pump_qmax

{100: 202.47, 80: 182.43, 60: 155.57, 40: 128.91, 20: 79.45}
Defining pump head with interpolation
measured_rpms = sorted(pump_data.keys())
measured_rpms

[20, 40, 60, 80, 100]
Implementing pump head calculation and interpolation
def interp_linear(x, x0, x1, y0, y1):
    if x1==x0: 
        return y0
    return y0 + (x-x0)/(x1-x0)*(y1-y0)

def pump_head(pump_rpm, flow_gpm):
    r=float(pump_rpm)
    q=float(flow_gpm)
    # clamp rpm to measured range
    if r <= measured_rpms[0]:
        r0=r1=measured_rpms[0]
    elif r >= measured_rpms[-1]:
        r0=r1=measured_rpms[-1]
    else:
        # find bounding
        for i in range(len(measured_rpms)-1):
            if measured_rpms[i] <= r <= measured_rpms[i+1]:
                r0=measured_rpms[i]; r1=measured_rpms[i+1]; break
    # interpolate qmax
    qmax0=pump_qmax[r0]; qmax1=pump_qmax[r1]
    qmax_r = interp_linear(r, r0
40 0.0
60 0.0
80 0.0
100 0.0
Processing system head functions with pchip interpolation
sys_funcs={}
sys_qmax={}
for zone, pts in sys_data.items():
    q=np.array([p[0] for p in pts])
    h=np.array([p[1] for p in pts])
    sys_qmax[zone]=q.max()
    f,_=pchip_interp(q,h)
    sys_funcs[zone]=f

sys_qmax

{'zone3': 134.24, 'zone2': 143.41, 'inlet': 180.09, 'zone1': 208.87}
Implementing bisection root-finding for pump system
def pump_qmax_at(pump_rpm):
    r=float(pump_rpm)
    if r<=measured_rpms[0]: return pump_qmax[measured_rpms[0]]
    if r>=measured_rpms[-1]: return pump_qmax[measured_rpms[-1]]
    for i in range(len(measured_rpms)-1):
        r0=measured_rpms[i]; r1=measured_rpms[i+1]
        if r0<=r<=r1:
            return interp_linear(r,r0,r1,pump_qmax[r0],pump_qmax[r1])
    return pump_qmax[measured_rpms[-1]]

def sys_head(zone, flow_gpm):
    q=float(flow_gpm)
    if q<0 or q>sys_qmax[zone]: 
        return np.nan
    return float(sys_funcs[zone](q))

def get_flow(pump_rpm, zone):
    qmax=min(pump_qmax_at(pump_rpm), sys_qmax[zone])
    # if qmax is 0:
    if qmax<=0: return 0.0
    # define g
    def g(q):
        hp=pump_head(pump_rpm,q)
        hs=sys_head(zone,q)
        if np.isnan(hp) or np.isnan(hs): 
            return np.nan
        return hp-hs
    # compute at low q
    qlo=0.0
    # use small positive to avoid NaN?
    qeps=min(0.1, qmax*1e-4 + 1e-4)
    glo=g(qeps)
    # if still nan, try another
    if np.isnan(glo):
        qeps=1e-3
        glo=g(qeps)
    # compute at high
    ghi=g(qmax)
    if np.isnan(ghi):
        # back off
        qhi=qmax
        for frac in [0.999,0.99,0.95,0.9,0.8]:
            qhi=qmax*frac
            ghi=g(qhi)
            if not np.isnan(ghi):
                qmax=qhi
                break
    # decide
    if np.isnan(glo) or np.isnan(ghi):
        return np.nan
    if glo<0:
        # pump can't overcome system even at near zero flow
        return 0.0
    if ghi>0:
        # pump above system even at max flow; operating point limited by curves endpoint
        return qmax
    # bracket and bisect
    a=qeps; b=qmax; fa=glo; fb=ghi
    # bisection
    for _ in range(60):
        m=0.5*(a+b)
        fm=g(m)
        if np.isnan(fm): 
            b=m; fb=fm
            continue
        if fm==0 or (b-a)<1e-6:
            return m
        if np.sign(fm)==np.sign(fa):
            a=m; fa=fm
        else:
            b=m; fb=fm
    return 0.5*(a+b)

# test get_flow at measured rpm maybe
for zone in sys_data:
    for r in [40,60,80,100]:
        q=get_flow(r,zone)
        print(zone,r,q)

zone3 40 58.35999966962262
zone3 60 86.060000194097
zone3 80 105.8099998693422
zone3 100 120.67999999827268
zone2 40 68.70999967623217
zone2 60 98.54000003302849
zone2 80 120.40000007955928
zone2 100 139.27999990470238
inlet 40 87.97000019793882
inlet 60 123.55999985054298
inlet 80 151.8299997223691
inlet 100 170.14999986769106
zone1 40 110.79000007611324
zone1 60 143.84000003810144
zone1 80 171.38000020755334
zone1 100 193.06999994566985
Computing pump power at various RPMs
def get_power(pump_rpm):
    motor_rpm=2.5*float(pump_rpm)
    x=motor_rpm
    y=7.2803e-6*((x+13.092)**3.18385)
    return float(y)

for r in [40,60,80,100]:
    print(r, "motor",2.5*r, "kW", get_power(r))

40 motor 100.0 kW 25.11703560326286
60 motor 150.0 kW 80.57549747254049
80 motor 200.0 kW 188.780782535984
100 motor 250.0 kW 369.3252436732171
Computing efficiency for different zones and rpms
zones=["inlet","zone1","zone2","zone3"]
rpm_grid=np.arange(40,101,1)
eff={}
flow={}
for z in zones:
    q_list=[]
    e_list=[]
    for r in rpm_grid:
        q=get_flow(r,z)
        p=get_power(r)
        q_list.append(q)
        e_list.append(q*60/p if p>0 else np.nan)
    flow[z]=np.array(q_list)
    eff[z]=np.array(e_list)

# find max efficiency and rpm
sweet={}
for z in zones:
    idx=np.nanargmax(eff[z])
    sweet[z]=(rpm_grid[idx], eff[z][idx], flow[z][idx], get_power(rpm_grid[idx]))
sweet

{'inlet': (40, 210.14422622352208, 87.97000019793882, 25.11703560326286),
 'zone1': (40, 264.65702838368617, 110.79000007611324, 25.11703560326286),
 'zone2': (40, 164.1356108138167, 68.70999967623217, 25.11703560326286),
 'zone3': (40, 139.41135552327987, 58.35999966962262, 25.11703560326286)}
Computing efficiency and flow for varying rpm
rpm_grid2=np.arange(20,101)
eff2={}
flow2={}
for z in zones:
    q_list=[]; e_list=[]
    for r in rpm_grid2:
        q=get_flow(r,z)
        p=get_power(r)
        q_list.append(q)
        e_list.append(q*60/p if p>0 else np.nan)
    eff2[z]=np.array(e_list); flow2[z]=np.array(q_list)

sweet2={}
for z in zones:
    idx=np.nanargmax(eff2[z])
    sweet2[z]=(rpm_grid2[idx], eff2[z][idx], flow2[z][idx], get_power(rpm_grid2[idx]))
sweet2

{'inlet': (20, 849.2887433454667, 55.450000092822336, 3.9173956226757736),
 'zone1': (20, 1096.034304536336, 71.55999978155214, 3.9173956226757736),
 'zone2': (20, 672.538662279776, 43.91000019491692, 3.9173956226757736),
 'zone3': (20, 574.667514554203, 37.52000010014335, 3.9173956226757736)}
Computing pump durations at different RPMs
volumes={"inlet":94000,"zone1":41000,"zone2":39000,"zone3":14000}
def duration_hours(zone, rpm):
    q=get_flow(rpm,zone)
    if q<=0 or np.isnan(q): 
        return np.inf
    return volumes[zone]/(q*60)

# durations at 20,40,60
for r in [20,30,40,50,60,70,80,90,100]:
    total=sum(duration_hours(z,r) for z in zones)
    print(r,total)

20 58.82468984633492
30 45.78235693266342
40 37.43514837229865
50 30.86144858529316
60 26.73764166425208
70 23.95304141118535
80 21.909680572899028
90 20.494708602658413
100 19.347211640525533
Evaluating pump rpm schedule for cost optimization
for z in zones:
    print(z, duration_hours(z,100))

inlet 9.20756196229744
zone1 3.53930353512003
zone2 4.666858130706063
zone3 1.9334880124019977
Computing energy per gallon for varying rpm and zones
def energy_per_gal(zone,rpm):
    q=get_flow(rpm,zone)
    p=get_power(rpm)
    if q<=0: return np.inf
    return p/(q*60)

for z in zones:
    print(z, "kWh/gal at 40", energy_per_gal(z,40), "at 100", energy_per_gal(z,100))

inlet kWh/gal at 40 0.004758636570563398 at 100 0.03617643686555049
zone1 kWh/gal at 40 0.003778475131029777 at 100 0.031881808305894724
zone2 kWh/gal at 40 0.006092523097466802 at 100 0.044194577341316696
zone3 kWh/gal at 40 0.0071730168338619525 at 100 0.051006137951400854
Computing energy cost for constant rpm schedules
prices={"off":0.074,"mid":0.102,"on":0.151}
def cost_for_constant_rpm(rpm, runtime):
    # runtime hours total, allocate to cheapest periods first
    off=min(12, runtime)
    rem=runtime-off
    mid=min(6, rem)
    rem-=mid
    on=min(6, rem)
    rem-=on
    if rem>1e-9:
        return np.inf
    P=get_power(rpm)
    return P*(off*prices["off"] + mid*prices["mid"] + on*prices["on"])

for r in [40,50,60,70,80,90,100]:
    T=sum(duration_hours(z,r) for z in zones)
    c=cost_for_constant_rpm(r,T)
    print(r, "T",T, "P",get_power(r), "cost",c)

40 T 37.43514837229865 P 25.11703560326286 cost inf
50 T 30.86144858529316 P 47.437830259939744 cost inf
60 T 26.73764166425208 P 80.57549747254049 cost inf
70 T 23.95304141118535 P 126.88293994639137 cost 304.38065869653263
80 T 21.909680572899028 P 188.780782535984 cost 394.6201300646355
90 T 20.494708602658413 P 268.7498417694583 cost 504.36309653021607
100 T 19.347211640525533 P 369.3252436732171 cost 629.1193148897152
Evaluating cost optimization with variable rpm and zones
for r in [70,80,90,100]:
    print("rpm",r)
    for z in zones:
        print(z, "Q", get_flow(r,z), "kWh/gal", energy_per_gal(z,r))
    print()

rpm 70
inlet Q 138.85372353978266 kWh/gal 0.015229808836687819
zone1 Q 158.07707320238086 kWh/gal 0.013377750630958285
zone2 Q 109.60983900121374 kWh/gal 0.01929311898496423
zone3 Q 96.52745439400323 kWh/gal 0.02190791914123622

rpm 80
inlet Q 151.8299997223691 kWh/gal 0.020722824088474145
zone1 Q 171.38000020755334 kWh/gal 0.018358888853946112
zone2 Q 120.40000007955928 kWh/gal 0.02613244496279614
zone3 Q 105.8099998693422 kWh/gal 0.029735813056279644

rpm 90
inlet Q 161.72238873572138 kWh/gal 0.02769662298774598
zone1 Q 182.567821451239 kWh/gal 0.02453424702056428
zone2 Q 129.86994615409378 kWh/gal 0.034489611816550206
zone3 Q 113.30064256022835 kWh/gal 0.03953343889563502

rpm 100
inlet Q 170.14999986769106 kWh/gal 0.03617643686555049
zone1 Q 193.06999994566985 kWh/gal 0.031881808305894724
zone2 Q 139.27999990470238 kWh/gal 0.044194577341316696
zone3 Q 120.67999999827268 kWh/gal 0.051006137951400854

Computing pump duration and power for zone1
for r in [40,50,60,65,70]:
    print(r, duration_hours("zone1", r), get_power(r))

40 6.167825010053978 25.11703560326286
50 5.337059489914393 47.437830259939744
60 4.750648867855443 80.57549747254049
65 4.5246814926374865 101.93319710802317
70 4.322785837883551 126.88293994639137
Calculating required pump rpm for zone1
target_flow=41000/(6*60)
target_flow

113.88888888888889
Calculating pump speed to meet target flow
# find rpm in [40,100] such that zone1 flow=target
def find_rpm_for_flow(zone, target, lo=40, hi=100):
    # monotonic flow increase with rpm likely
    a=lo; b=hi
    fa=get_flow(a,zone)-target
    fb=get_flow(b,zone)-target
    if fa>0: return lo
    if fb<0: return hi
    for _ in range(60):
        m=0.5*(a+b)
        fm=get_flow(m,zone)-target
        if abs(fm) < 1e-6:
            return m
        if np.sign(fm)==np.sign(fa):
            a=m; fa=fm
        else:
            b=m; fb=fm
    return 0.5*(a+b)

rpm_needed=find_rpm_for_flow("zone1", target_flow, 40, 80)
rpm_needed, get_flow(rpm_needed,"zone1")

(41.71398639678955, 113.88888834614514)
Computing power increase with rpm
get_power(rpm_needed)

28.274226100080995
Optimizing pumping schedule for cost minimization
import scipy, scipy.optimize, scipy.interpolate, scipy.stats

Setting up linear programming variables and constraints
from scipy.optimize import linprog
rpms = np.arange(40,101)  # integer
periods = ["off","mid","on"]
hours_available = {"off":12.0,"mid":6.0,"on":6.0}
price = {"off":0.074,"mid":0.102,"on":0.151}

# build variable indexing
var_index=[]
for p in periods:
    for z in zones:
        for r in rpms:
            var_index.append((p,z,int(r)))
n=len(var_index)
n

732
Optimizing pump usage with constraints
# cost vector
c=np.zeros(n)
Qmat=np.zeros((len(zones), n))  # gallons per hour per variable?
# Actually volume = Q*60* t; so coefficient = Q*60.
for j,(p,z,r) in enumerate(var_index):
    c[j]=get_power(r)*price[p]
    Qmat[zones.index(z),j]=get_flow(r,z)*60

# Equality constraints: each zone volume
A_eq=Qmat
b_eq=np.array([volumes[z] for z in zones],dtype=float)

# Inequality constraints: period time limits
A_ub=np.zeros((len(periods), n))
b_ub=np.array([hours_available[p] for p in periods],dtype=float)
for i,p in enumerate(periods):
    for j,(pp,z,r) in enumerate(var_index):
        if pp==p:
            A_ub[i,j]=1.0

bounds=[(0,None)]*n
res=linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
res.status, res.message, res.fun

(0,
 'Optimization terminated successfully. (HiGHS Status 7: Optimal)',
 282.6691320063784)
Inspecting RPM usage in LP solution
x=res.x
# Extract non-zero vars
nz=[(var_index[i], x[i]) for i in range(n) if x[i]>1e-6]
len(nz), nz[:10]

(7,
 [(('off', 'inlet', 76), 5.052709729686284),
  (('off', 'zone2', 78), 4.675458829719294),
  (('off', 'zone3', 76), 1.0689201525961045),
  (('off', 'zone3', 77), 1.2029112879982709),
  (('mid', 'inlet', 69), 6.000000000000013),
  (('on', 'zone1', 55), 5.015114719749861),
  (('on', 'zone2', 60), 0.9848852802501435)])
Evaluating schedule feasibility and cost optimization
periods5=[("off1","off",7.0),("mid1","mid",4.0),("on","on",6.0),("mid2","mid",2.0),("off2","off",5.0)]
prices5={"off1":prices["off"],"off2":prices["off"],"mid1":prices["mid"],"mid2":prices["mid"],"on":prices["on"]}
hours5={name:dur for name,_,dur in periods5}

var_index5=[]
for p,_,dur in periods5:
    for z in zones:
        for r in rpms:
            var_index5.append((p,z,int(r)))
n5=len(var_index5)
c5=np.zeros(n5)
Qmat5=np.zeros((len(zones), n5))
for j,(p,z,r) in enumerate(var_index5):
    c5[j]=get_power(r)*prices5[p]
    Qmat5[zones.index(z),j]=get_flow(r,z)*60
A_eq=Qmat5
b_eq=b_eq
A_ub=np.zeros((len(periods5), n5))
b_ub=np.array([hours5[p] for p,_,_ in periods5])
for i,(p,_,dur) in enumerate(periods5):
    for j,(pp,z,r) in enumerate(var_index5):
        if pp==p:
            A_ub[i,j]=1
res5=linprog(c5
(0, 282.66913200637833)
Viewing nonzero variables
x=res5.x
nz5=[(var_index5[i],x[i]) for i in range(n5) if x[i]>1e-6]
len(nz5), nz5

(9,
 [(('off1', 'inlet', 76), 5.052709729686318),
  (('off1', 'zone3', 76), 0.7443789823155638),
  (('off1', 'zone3', 77), 1.202911287998137),
  (('mid1', 'inlet', 69), 3.999999999999996),
  (('on', 'zone1', 55), 5.015114719749886),
  (('on', 'zone2', 60), 0.9848852802501089),
  (('mid2', 'inlet', 69), 1.9999999999999907),
  (('off2', 'zone2', 78), 4.675458829719284),
  (('off2', 'zone3', 76), 0.32454117028069823)])
Evaluating optimization with zone1 onpeak constraint
# Constrain on period to only zone1 variables by setting others bounds to 0
bounds = []
for (p,z,r) in var_index5:
    if p=="on" and z!="zone1":
        bounds.append((0,0))
    else:
        bounds.append((0,None))
res_onzone1 = linprog(c5, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
res_onzone1.status, res_onzone1.fun

(0, 300.27102227358455)
Computing optimal pumping schedule and volumes
# compute volumes for this schedule as per LP durations and rpms
plan = [
("off1","inlet",76,5.052709729686318),
("off1","zone3",77,1.202911287998137),
("off1","zone3",76,0.7443789823155638),
("mid1","inlet",69,4.0),
("on","zone1",55,5.015114719749886),
("on","zone2",60,0.9848852802501089),
("mid2","inlet",69,2.0),
("off2","zone2",78,4.675458829719284),
("off2","zone3",76,0.32454117028069823)
]
delivered={z:0.0 for z in zones}
for p,z,r,t in plan:
    delivered[z]+=get_flow(r,z)*60*t
delivered, volumes

({'inlet': 93999.99999999991,
  'zone1': 41000.00000000003,
  'zone2': 38999.99999999999,
  'zone3': 14000.00000000002},
 {'inlet': 94000, 'zone1': 41000, 'zone2': 39000, 'zone3': 14000})
Scheduling time segments to daily periods
from datetime import datetime, timedelta, time

def h_to_hm(hours):
    mins=hours*60
    h=int(mins//60)
    m=int(round(mins%60))
    if m==60: h+=1; m=0
    return h,m

def add_time(t, hours):
    return t+timedelta(seconds=hours*3600)

start=datetime(2025,11,30,0,0,0)  # date irrelevant
t=start
sched=[]
# create sequential schedule over 5 periods in actual chronology, using plan pieces in that order.
def format_dt(dt):
    return dt.strftime("%H:%M")
# We'll build within each period
period_order=["off1","mid1","on","mid2","off2"]
current=start
# Build mapping p->list of segments in desired order
segments_by_p={p:[] for p in
{'off1': [('inlet', 76, 5.052709729686318),
  ('zone3', 77, 1.202911287998137),
  ('zone3', 76, 0.7443789823155638)],
 'mid1': [('inlet', 69, 4.0)],
 'on': [('zone1', 55, 5.015114719749886), ('zone2', 60, 0.9848852802501089)],
 'mid2': [('inlet', 69, 2.0)],
 'off2': [('zone2', 78, 4.675458829719284),
  ('zone3', 76, 0.32454117028069823)]}
Generating schedule with start and end times
period_starts={
"off1": datetime(2025,11,30,0,0,0),
"mid1": datetime(2025,11,30,7,0,0),
"on": datetime(2025,11,30,11,0,0),
"mid2": datetime(2025,11,30,17,0,0),
"off2": datetime(2025,11,30,19,0,0),
}
period_durs={"off1":7,"mid1":4,"on":6,"mid2":2,"off2":5}
schedule=[]
for p in period_order:
    t=period_starts[p]
    for z,r,hours in segments_by_p[p]:
        end=add_time(t,hours)
        schedule.append((t,end,p,z,r,hours))
        t=end
    # maybe check ends at period start+dur
    assert abs((t-(period_starts[p]+timedelta(hours=period_durs[p]))).total_seconds())<1e-6
schedule[:3], schedule[-2:]

([(datetime.datetime(2025, 11, 30, 0, 0),
   datetime.datetime(2025, 11, 30, 5, 3, 9, 755027),
   'off1',
   'inlet',
   76,
   5.052709729686318),
  (datetime.datetime(2025, 11, 30, 5, 3, 9, 755027),
   datetime.datetime(2025, 11, 30, 6, 15, 20, 235664),
   'off1',
   'zone3',
   77,
   1.202911287998137),
  (datetime.datetime(2025, 11, 30, 6, 15, 20, 235664),
   datetime.datetime(2025, 11, 30, 7, 0),
   'off1',
   'zone3',
   76,
   0.7443789823155638)],
 [(datetime.datetime(2025, 11, 30, 19, 0),
   datetime.datetime(2025, 11, 30, 23, 40, 31, 651787),
   'off2',
   'zone2',
   78,
   4.675458829719284),
  (datetime.datetime(2025, 11, 30, 23, 40, 31, 651787),
   datetime.datetime(2025, 12, 1, 0, 0),
   'off2',
   'zone3',
   76,
   0.32454117028069823)])
Computing per-segment volume and cost
rows=[]
for start,end,p,z,r,hours in schedule:
    Q=get_flow(r,z)
    P=get_power(r)
    vol=Q*60*hours
    energy=P*hours
    cost=energy*prices5[p]
    rows.append({
        "Start": start.strftime("%H:%M"),
        "End": end.strftime("%H:%M"),
        "Period": p,
        "Zone": z,
        "Pump RPM": r,
        "Motor RPM": 2.5*r,
        "Flow (GPM)": Q,
        "Power (kW)": P,
        "Hours": hours,
        "Volume (gal)": vol,
        "Energy (kWh)": energy,
        "Cost ($)": cost
    })
df=pd.DataFrame(rows)
df, df["Volume (gal)"].sum(), df["Cost ($)"].sum()

(   Start    End Period   Zone  Pump RPM  Motor RPM  Flow (GPM)  Power (kW)  \
 0  00:00  05:03   off1  inlet        76      190.0  146.873405  161.993299   
 1  05:03  06:15   off1  zone3        77      192.5  103.134339  168.427926   
 2  06:15  07:00   off1  zone3        76      190.0  102.226413  161.993299   
 3  07:00  11:00   mid1  inlet        69      172.5  137.426331  121.591061   
 4  11:00  16:00     on  zone1        55      137.5  136.254776   62.509188   
 5  16:00  17:00     on  zone2        60      150.0   98.540000   80.575497   
 6  17:00  19:00   mid2  inlet        69      172.5  137.426331  121.591061   
 7  19:00  23:40   off2  zone2        78      195.0  118.266340  175.035718   
 8  23:40  00:00   off2  zone3        76      190.0  102.226413  161.993299   
 
       Hours  Volume (gal)  Energy (kWh)   Cost ($)  
 0  5.052710  44526.520802    818.505117  60.569379  
 1  1.202911   7443.687633    202.603854  14.992685  
 2  0.744379   4565.711588    120.584407   8.923246  
 3  4.000000  32982.319465    486.364246  49.609153  
 4  5.015115  41000.000000    313.490750  47.337103  
 5  0.984885   5823.035733     79.357621  11.983001  
 6  2.000000  16491.159733    243.182123  24.804577  
 7  4.675459  33176.964267    818.372294  60.559550  
 8  0.324541   1990.600779     52.573495   3.890439  ,
 187999.99999999997,
 282.66913200637856)
Calculating minimum RPM for inlet duration
def find_min_rpm_for_duration(zone, max_hours, lo=40, hi=100):
    # find smallest rpm such that duration<=max_hours
    a=lo; b=hi
    if duration_hours(zone,lo) <= max_hours: return lo
    if duration_hours(zone,hi) > max_hours: return None
    # monotonic decreasing duration; bisection
    for _ in range(60):
        m=0.5*(a+b)
        if duration_hours(zone,m) <= max_hours:
            b=m
        else:
            a=m
    return b

minrpm_inlet12=find_min_rpm_for_duration("inlet",12,40,100)
minrpm_inlet12, duration_hours("inlet",minrpm_inlet12), get_flow(minrpm_inlet12,"inlet")

(64.43962558328934, 11.999999999999998, 130.55555555555557)
Determining optimal rpm for total duration
# solve for rpm such that total duration=24 using get_flow functions
def total_duration(r):
    return sum(duration_hours(z,r) for z in zones)
# bisection between 60 and 70 because at 60 is 26.7>24; at 70 23.95<24
a=60; b=70
for _ in range(60):
    m=0.5*(a+b)
    if total_duration(m) > 24:
        a=m
    else:
        b=m
b, total_duration(b)

(69.8023726159847, 24.0)
Calculating energy cost at constant rpm
rpm_const=69.8023726159847
P=get_power(rpm_const)
cost_const=P*(12*prices["off"]+6*prices["mid"]+6*prices["on"])
P, cost_const

(125.82484047144317, 302.73456617429224)
Calculating zone cutoff rpm for positive flow
# determine cutoff rpm for each zone where flow becomes positive
def cutoff_rpm(zone, rpm_min=20, rpm_max=100, step=0.1):
    r=rpm_min
    while r<=rpm_max:
        q=get_flow(r, zone)
        if q>1e-6:
            return r
        r+=step
    return None
for z in zones:
    print(z, cutoff_rpm(z,20,100,0.1))

inlet 20
zone1 20
zone2 20
zone3 20
Generating schedule with minute-level resolution and adjustments
plan_minutes=[
("off1","inlet",76,303),
("off1","zone3",77,72),
("off1","zone3",76,45),
("mid1","inlet",69,240),
("on","zone1",55,301),
("on","zone2",60,59),
("mid2","inlet",69,120),
("off2","zone2",78,281),
("off2","zone3",76,19),
]
delivered_m={z:0.0 for z in zones}
for p,z,r,mins in plan_minutes:
    delivered_m[z]+=get_flow(r,z)*mins
delivered_m, {z:delivered_m[z]-volumes[z] for z in zones}

({'inlet': 93976.12076653872,
  'zone1': 41012.68761078139,
  'zone2': 39046.70159243609,
  'zone3': 13968.162827227989},
 {'inlet': -23.87923346128082,
  'zone1': 12.687610781387775,
  'zone2': 46.70159243608941,
  'zone3': -31.8371727720114})
Computing flow for zone1 at varying rpms
get_flow(55,"zone1"), get_flow(54,"zone1"), (get_flow(55,"zone1")-get_flow(54,"zone1"))

(136.25477611555277, 134.6562754474434, 1.5985006681093807)
Optimizing flow to match target volume
def zone1_volume_mix(min55, min54):
    return get_flow(55,"zone1")*min55 + get_flow(54,"zone1")*min54
target=volumes["zone1"]
best=None
for m54 in range(0,61): # up to an hour
    m55=301-m54
    vol=zone1_volume_mix(m55,m54)
    err=vol-target
    if err<=0 and abs(err)<=100: # no oversupply, under within 100
        best=(m55,m54,vol,err)
        break
best

(293, 8, 40999.89960543651, -0.10039456348749809)
Calculating flow difference for zone2 at different rpms
get_flow(78,"zone2"),get_flow(77,"zone2"), get_flow(78,"zone2")-get_flow(77,"zone2")

(118.2663401796705, 117.19469183516134, 1.071648344509157)
Comparing flow rates at different RPMs
for r in [78,77,76,75,74]:
    print(r, get_flow(r,"zone2"))

78 118.2663401796705
77 117.19469183516134
76 116.11991153418506
75 115.04206925564282
74 113.96124245709689
Computing zone2 flow allocation for target volume
Q60=get_flow(60,"zone2")
Q78=get_flow(78,"zone2")
Q76=get_flow(76,"zone2")
vol_on=Q60*59
target_off=39000-vol_on
target_off

33186.139998051316
Solving for optimal rpm mix to meet target
# need m78*Q78 + (281-m78)*Q76 <= target_off, choose integer m78
best=None
for m78 in range(0,282):
    vol_off=m78*Q78 + (281-m78)*Q76
    total=vol_on+vol_off
    err=total-39000
    if err<=0 and abs(err)<=100:
        best=(m78,281-m78,total,err)
        break
best

(213, 68, 38900.74444454308, -99.25555545691896)
Searching optimal rpm combination for zone2
Q77=get_flow(77,"zone2")
best=None
best_abs=1e9
best_pref=None
for m78 in range(0,282):
    for m77 in range(0,282-m78):
        m76=281-m78-m77
        vol_off=m78*Q78 + m77*Q77 + m76*Q76
        total=vol_on+vol_off
        err=total-39000
        if err<=0 and abs(err)<=100:
            if abs(err)<best_abs:
                best_abs=abs(err)
                best=(m78,m77,m76,total,err)
best

(237, 44, 0, 38999.54906527769, -0.45093472230655607)
Computing zone3 flow difference at different RPMs
get_flow(77,"zone3"),get_flow(76,"zone3"), get_flow(77,"zone3")-get_flow(76,"zone3")

(103.13433898734178, 102.22641281467783, 0.9079261726639487)
Computing new schedule and periods totals
# Build new minute plan with above modifications, ensure period lengths satisfied.
minute_plan2=[
("off1","inlet",76,303),
("off1","zone3",77,72),
("off1","zone3",76,45),
("mid1","inlet",69,240),
("on","zone1",55,293),
("on","zone1",54,8),
("on","zone2",60,59),
("mid2","inlet",69,120),
("off2","zone2",78,237),
("off2","zone2",77,44),
("off2","zone3",76,19),
]

# check period totals
from collections import defaultdict
tot=defaultdict(int)
for p,z,r,m in minute_plan2:
    tot[p]+=m
tot

defaultdict(int,
            {'off1': 420, 'mid1': 240, 'on': 360, 'mid2': 120, 'off2': 300})
Computing volumes, costs, and energy usage
delivered2={z:0.0 for z in zones}
cost2=0.0
energy2=0.0
for p,z,r,m in minute_plan2:
    hours=m/60
    delivered2[z]+=get_flow(r,z)*m
    E=get_power(r)*hours
    energy2+=E
    cost2+=E*prices5[p]
delivered2, {z:delivered2[z]-volumes[z] for z in zones}, cost2, energy2

({'inlet': 93976.12076653872,
  'zone1': 40999.89960543651,
  'zone2': 38999.549065277686,
  'zone3': 13968.162827227989},
 {'inlet': -23.87923346128082,
  'zone1': -0.10039456348749809,
  'zone2': -0.450934722313832,
  'zone3': -31.8371727720114},
 282.24720741027835,
 3129.8115297295994)
Computing time schedule with minute resolution
# convert minute plan into actual time schedule
period_starts_m = {
"off1": datetime(2025,11,30,0,0),
"mid1": datetime(2025,11,30,7,0),
"on": datetime(2025,11,30,11,0),
"mid2": datetime(2025,11,30,17,0),
"off2": datetime(2025,11,30,19,0),
}
period_order=["off1","mid1","on","mid2","off2"]
segments_by_p_m={p:[] for p in period_order}
for p,z,r,m in minute_plan2:
    segments_by_p_m[p].append((z,r,m))
# build schedule ordered by those lists in plan order
schedule_m=[]
for p in period_order:
    t=period_starts_m[p]
    for z,r,m in segments_by_p_m[p]:
        end=t+timedelta(minutes=m)
        schedule_m.append((t,end,p,z,r,m))
        t=end
schedule_m[-1]

(datetime.datetime(2025, 11, 30, 23, 41),
 datetime.datetime(2025, 12, 1, 0, 0),
 'off2',
 'zone3',
 76,
 19)
Creating energy usage and cost table
rows=[]
for start,end,p,z,r,m in schedule_m:
    Q=get_flow(r,z)
    P=get_power(r)
    vol=Q*m
    hours=m/60
    energy=P*hours
    cost=energy*prices5[p]
    rows.append([start.strftime("%H:%M"), end.strftime("%H:%M"), p, z, r, int(2.5*r), Q, P, m, vol, energy, cost])
df_m=pd.DataFrame(rows, columns=["Start","End","TOU","Zone","Pump RPM","Motor RPM","Flow (GPM)","Power (kW)","Minutes","Volume(gal)","Energy(kWh)","Cost($)"])
df_m

    Start    End   TOU   Zone  Pump RPM  Motor RPM  Flow (GPM)  Power (kW)  \
0   00:00  05:03  off1  inlet        76        190  146.873405  161.993299   
1   05:03  06:15  off1  zone3        77        192  103.134339  168.427926   
2   06:15  07:00  off1  zone3        76        190  102.226413  161.993299   
3   07:00  11:00  mid1  inlet        69        172  137.426331  121.591061   
4   11:00  15:53    on  zone1        55        137  136.254776   62.509188   
5   15:53  16:01    on  zone1        54        135  134.656275   59.264729   
6   16:01  17:00    on  zone2        60        150   98.540000   80.575497   
7   17:00  19:00  mid2  inlet        69        172  137.426331  121.591061   
8   19:00  22:57  off2  zone2        78        195  118.266340  175.035718   
9   22:57  23:41  off2  zone2        77        192  117.194692  168.427926   
10  23:41  00:00  off2  zone3        76        190  102.226413  161.993299   

    Minutes   Volume(gal)  Energy(kWh)    Cost($)  
0       303  44502.641569   818.066159  60.536896  
1        72   7425.672407   202.113512  14.956400  
2        45   4600.188577   121.494974   8.990628  
3       240  32982.319465   486.364246  49.609153  
4       293  39922.649402   305.253202  46.093234  
5         8   1077.250204     7.901964   1.193197  
6        59   5813.860002    79.232573  11.964118  
7       120  16491.159733   243.182123  24.804577  
8       237  28029.122623   691.391087  51.162940  
9        44   5156.566441   123.513813   9.140022  
10       19   1942.301843    51.297878   3.796043  
Computing totals by zone and overall
df_m.groupby("Zone")["Volume(gal)"].sum(), df_m["Cost($)"].sum(), df_m["Energy(kWh)"].sum()

(Zone
 inlet    93976.120767
 zone1    40999.899605
 zone2    38999.549065
 zone3    13968.162827
 Name: Volume(gal), dtype: float64,
 282.24720741027835,
 3129.8115297296)
Calculating cost and energy breakdown by period
df_m.groupby("TOU")["Cost($)"].sum(), df_m.groupby("TOU")["Energy(kWh)"].sum()

(TOU
 mid1    49.609153
 mid2    24.804577
 off1    84.483924
 off2    64.099006
 on      59.250549
 Name: Cost($), dtype: float64,
 TOU
 mid1     486.364246
 mid2     243.182123
 off1    1141.674645
 off2     866.202778
 on       392.387739
 Name: Energy(kWh), dtype: float64)
Thought for 13m 32s
Done
