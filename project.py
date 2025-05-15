import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ProjectileMotionCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("포물선 운동 계산기")
        
        # 왼쪽 프레임 (입력 + 테이블)
        left_frame = ttk.Frame(root)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)
        
        # 입력 프레임
        input_frame = ttk.LabelFrame(left_frame, text="입력 값", padding="10")
        input_frame.grid(row=0, column=0, sticky="nsew")

        # 입력 필드들
        ttk.Label(input_frame, text="초기 높이 (m):").grid(row=0, column=0, padx=5, pady=5)
        self.initial_height = ttk.Entry(input_frame, width=15)
        self.initial_height.grid(row=0, column=1, padx=5, pady=5)
        self.initial_height.insert(0, "0")
        
        ttk.Label(input_frame, text="초기 속도 (m/s):").grid(row=1, column=0, padx=5, pady=5)
        self.initial_velocity = ttk.Entry(input_frame, width=15)
        self.initial_velocity.grid(row=1, column=1, padx=5, pady=5)
        self.initial_velocity.insert(0, "10")
        
        ttk.Label(input_frame, text="발사 각도 (도):").grid(row=2, column=0, padx=5, pady=5)
        self.angle = ttk.Entry(input_frame, width=15)
        self.angle.grid(row=2, column=1, padx=5, pady=5)
        self.angle.insert(0, "45")
        
        ttk.Label(input_frame, text="X축 가속도 (m/s²):").grid(row=3, column=0, padx=5, pady=5)
        self.ax_input = ttk.Entry(input_frame, width=15)
        self.ax_input.grid(row=3, column=1, padx=5, pady=5)
        self.ax_input.insert(0, "0")
        
        ttk.Label(input_frame, text="Y축 가속도 (m/s²):").grid(row=4, column=0, padx=5, pady=5)
        self.ay_input = ttk.Entry(input_frame, width=15)
        self.ay_input.grid(row=4, column=1, padx=5, pady=5)
        self.ay_input.insert(0, "-9.81")
        
        # 계산 버튼
        ttk.Button(input_frame, text="계산하기", command=self.calculate).grid(row=5, column=0, columnspan=2, pady=10)
        
        # 특성점 결과 프레임
        self.result_frame = ttk.LabelFrame(left_frame, text="운동 특성점", padding="10")
        self.result_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # 특성점 결과 레이블
        self.result_text = ttk.Label(self.result_frame, text="계산 전", justify="left", wraplength=300)
        self.result_text.grid(row=0, column=0, padx=5, pady=5)

        # 시간 검색 프레임을 table_frame 이전으로 이동
        search_frame = ttk.LabelFrame(left_frame, text="특정 시간 검색", padding="10")
        search_frame.grid(row=2, column=0, sticky="nsew", pady=10)
        
        ttk.Label(search_frame, text="시간 (s):").grid(row=0, column=0, padx=5, pady=5)
        self.search_time = ttk.Entry(search_frame, width=15)
        self.search_time.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(search_frame, text="검색", command=self.search_time_data).grid(row=0, column=2, padx=5, pady=5)
        
        # 검색 결과 표시 레이블
        self.search_result = ttk.Label(search_frame, text="", wraplength=300)
        self.search_result.grid(row=1, column=0, columnspan=3, pady=5)
        
        # 데이터 테이블 프레임의 위치를 변경
        self.table_frame = ttk.LabelFrame(left_frame, text="운동 데이터", padding="10")
        self.table_frame.grid(row=3, column=0, sticky="nsew", pady=10)  # row를 3으로 변경
        
        # 오른쪽 프레임 (그래프)
        self.graph_frame = ttk.Frame(root)
        self.graph_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)
        
        # 데이터 저장 변수
        self.current_data = None

    def search_time_data(self):
        if self.current_data is None:
            messagebox.showwarning("경고", "먼저 계산을 실행해주세요.")
            return
            
        try:
            search_time = float(self.search_time.get())
            t, x, y, vx, vy, ax, ay = self.current_data
            
            # 가장 가까운 시간 인덱스 찾기
            closest_idx = np.abs(t - search_time).argmin()
            
            # 전체 변위 계산 (처음 위치로부터의 변화)
            dx = x[closest_idx] - x[0]
            dy = y[closest_idx] - y[0]
            total_displacement = np.sqrt(dx**2 + dy**2)  # 전체 변위의 크기
            
            # 속도 계산 (벡터의 크기)
            total_velocity = np.sqrt(vx[closest_idx]**2 + vy[closest_idx]**2)
            
            result_text = f"시간 {search_time}초일 때:\n"
            result_text += f"X 위치: {x[closest_idx]:.3f} m\n"
            result_text += f"Y 위치: {y[closest_idx]:.3f} m\n"
            result_text += f"X 변위: {dx:.3f} m\n"
            result_text += f"Y 변위: {dy:.3f} m\n"
            result_text += f"전체 변위: {total_displacement:.3f} m\n"
            result_text += f"X 속도: {vx[closest_idx]:.3f} m/s\n"
            result_text += f"Y 속도: {vy[closest_idx]:.3f} m/s\n"
            result_text += f"전체 속도: {total_velocity:.3f} m/s\n"
            result_text += f"X 가속도: {ax[closest_idx]:.3f} m/s²\n"
            result_text += f"Y 가속도: {ay[closest_idx]:.3f} m/s²"
            
            self.search_result.config(text=result_text)
            
        except ValueError:
            messagebox.showerror("오류", "올바른 시간 값을 입력해주세요.")

    def calculate_characteristics(self, t, x, y, vx, vy, ax, ay):
        # 최고점 찾기
        max_height_idx = np.argmax(y)
        max_height = y[max_height_idx]
        time_to_max = t[max_height_idx]
        
        # 최고점에서의 속도
        v_at_max = np.sqrt(vx[max_height_idx]**2 + vy[max_height_idx]**2)
        
        # 도달 거리 찾기 (y가 0이 되거나 음수가 되는 지점들)
        ground_indices = np.where(y <= 0)[0]
        
        if len(ground_indices) > 0:
            # 시작점이 y=0인 경우, 첫 번째 교차점을 건너뛰고 두 번째 교차점을 찾음
            if abs(y[0]) < 1e-10:  # 부동소수점 오차를 고려하여 0 체크
                if len(ground_indices) > 1:
                    # 시작점 이후의 첫 번째 양수 y값을 찾음
                    positive_indices = np.where(y > 0)[0]
                    if len(positive_indices) > 0:
                        first_positive = positive_indices[0]
                        # 그 이후의 지면 교차점을 찾음
                        later_grounds = ground_indices[ground_indices > first_positive]
                        if len(later_grounds) > 0:
                            range_idx = later_grounds[0]
                        else:
                            range_idx = -1
                    else:
                        range_idx = -1
                else:
                    range_idx = -1
            else:
                range_idx = ground_indices[0]
                
            if range_idx > 0:  # 보간을 통한 정확한 도달 거리 계산
                t1, t2 = t[range_idx-1], t[range_idx]
                y1, y2 = y[range_idx-1], y[range_idx]
                x1, x2 = x[range_idx-1], x[range_idx]
                
                # 선형 보간으로 정확한 지면 도달 시간 계산
                t_ground = t1 + (t2 - t1) * (-y1)/(y2 - y1)
                range_distance = x1 + (x2 - x1) * (-y1)/(y2 - y1)
                time_of_flight = t_ground
            else:
                range_distance = x[-1]
                time_of_flight = t[-1]
        else:
            range_distance = x[-1]
            time_of_flight = t[-1]
        
        result_str = f"""최고점 특성:
        • 최고 높이: {max_height:.3f} m
        • 최고점 도달 시간: {time_to_max:.3f} s
        • 최고점에서의 속도: {v_at_max:.3f} m/s
        • 수평 도달 거리: {range_distance:.3f} m"""
        
        return result_str

    def create_data_table(self, t, x, y, vx, vy, ax, ay):
        # 1초 간격의 데이터 포인트 선택
        t_max = int(np.floor(t[-1]))  # 최대 시간을 정수로 내림
        t_points = np.arange(0, t_max + 1)  # 0초부터 최대 시간까지 1초 간격
        
        # 각 시간에 해당하는 데이터 찾기
        indices = [np.abs(t - tp).argmin() for tp in t_points]
        
        # X축 데이터
        x_selected = x[indices]
        vx_selected = vx[indices]
        dx = np.diff(x_selected)  # 구간 거리
        dx = np.append(dx, dx[-1] if len(dx) > 0 else 0)  # 마지막 값 처리
        
        # Y축 데이터
        y_selected = y[indices]
        vy_selected = vy[indices]
        dy = np.diff(y_selected)  # 구간 변위
        dy = np.append(dy, dy[-1] if len(dy) > 0 else 0)  # 마지막 값 처리
        
        # X축 데이터 테이블
        x_data = pd.DataFrame({
            '시간(s)': t_points,
            'X위치(m)': np.round(x_selected, 3),
            'X구간거리(m)': np.round(dx, 3),
            'X속도(m/s)': np.round(vx_selected, 3)
        })
        
        # Y축 데이터 테이블
        y_data = pd.DataFrame({
            '시간(s)': t_points,
            'Y위치(m)': np.round(y_selected, 3),
            'Y구간변위(m)': np.round(dy, 3),
            'Y속도(m/s)': np.round(vy_selected, 3),
            'Y가속도(m/s²)': np.round(ay[indices], 3)
        })
        
        # 이전 테이블 제거
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        # X축 테이블
        ttk.Label(self.table_frame, text="X축 운동 데이터:", font=('Malgun Gothic', 10, 'bold')).grid(row=0, column=0, pady=5)
        x_tree = ttk.Treeview(self.table_frame, columns=list(x_data.columns), show='headings', height=5)
        x_scroll = ttk.Scrollbar(self.table_frame, orient="vertical", command=x_tree.yview)
        x_tree.configure(yscrollcommand=x_scroll.set)
        
        for col in x_data.columns:
            x_tree.heading(col, text=col)
            x_tree.column(col, width=100)
        for i, row in x_data.iterrows():
            x_tree.insert('', 'end', values=list(row))
        
        x_tree.grid(row=1, column=0, padx=5, pady=5)
        x_scroll.grid(row=1, column=1, sticky='ns')
        
        # Y축 테이블
        ttk.Label(self.table_frame, text="Y축 운동 데이터:", font=('Malgun Gothic', 10, 'bold')).grid(row=2, column=0, pady=5)
        y_tree = ttk.Treeview(self.table_frame, columns=list(y_data.columns), show='headings', height=5)
        y_scroll = ttk.Scrollbar(self.table_frame, orient="vertical", command=y_tree.yview)
        y_tree.configure(yscrollcommand=y_scroll.set)
        
        for col in y_data.columns:
            y_tree.heading(col, text=col)
            y_tree.column(col, width=100)
        for i, row in y_data.iterrows():
            y_tree.insert('', 'end', values=list(row))
        
        y_tree.grid(row=3, column=0, padx=5, pady=5)
        y_scroll.grid(row=3, column=1, sticky='ns')

    def calculate(self):
        # 입력값 가져오기
        h0 = float(self.initial_height.get())
        v0 = float(self.initial_velocity.get())
        theta = float(self.angle.get())
        ax = float(self.ax_input.get())
        ay = float(self.ay_input.get())
        
        # 라디안으로 변환
        theta_rad = np.radians(theta)
        
        # 초기 속도 성분
        v0x = v0 * np.cos(theta_rad)
        v0y = v0 * np.sin(theta_rad)
        
        # 비행 시간 계산
        if ay == 0:
            if v0y == 0:
                t_flight = abs(2 * v0x)
            else:
                t_flight = abs(2 * v0y / 0.001)
        else:
            # 이차방정식: (1/2)ay*t^2 + v0y*t + h0 = 0
            a = 0.5 * ay
            b = v0y
            c = h0
            discriminant = b**2 - 4*a*c
            
            if discriminant >= 0:
                t1 = (-b + np.sqrt(discriminant)) / (2*a)
                t2 = (-b - np.sqrt(discriminant)) / (2*a)
                t_flight = max(t1, t2) if t1 > 0 or t2 > 0 else 10
            else:
                t_flight = abs(2 * v0y / ay)
        
        # 시간 배열 생성
        t = np.linspace(0, t_flight, 1000)  # 데이터 포인트 증가
        
        # 위치 계산
        x = v0x*t + 0.5*ax*t**2
        y = h0 + v0y*t + 0.5*ay*t**2
        
        # 속도 계산
        vx = v0x + ax*t
        vy = v0y + ay*t
        v = np.sqrt(vx**2 + vy**2)
        
        # 가속도
        ax_array = ax * np.ones_like(t)
        ay_array = ay * np.ones_like(t)
        
        # 현재 데이터 저장
        self.current_data = (t, x, y, vx, vy, ax_array, ay_array)

        # 특성점 계산 및 표시
        result_str = self.calculate_characteristics(t, x, y, vx, vy, ax_array, ay_array)
        self.result_text.config(text=result_str)
        
        # 그래프 생성
        fig = plt.Figure(figsize=(10, 8))
        fig.subplots_adjust(hspace=0.4)
        
        # 포물선 궤적
        ax1 = fig.add_subplot(221)
        ax1.plot(x, y)
        ax1.set_title('포물선 궤적')
        ax1.set_xlabel('거리 (m)')
        ax1.set_ylabel('높이 (m)')
        ax1.grid(True)
        
        # 시간-속도 그래프
        ax2 = fig.add_subplot(222)
        ax2.plot(t, v)
        ax2.set_title('시간-속도 그래프')
        ax2.set_xlabel('시간 (s)')
        ax2.set_ylabel('속도 (m/s)')
        ax2.grid(True)

        # 시간-변위 그래프
        ax3 = fig.add_subplot(223)
        ax3.plot(t, x, label='x-방향')
        ax3.plot(t, y, label='y-방향')
        ax3.set_title('시간-변위 그래프')
        ax3.set_xlabel('시간 (s)')
        ax3.set_ylabel('변위 (m)')
        ax3.legend()
        ax3.grid(True)
        
        # 시간-가속도 그래프
        ax4 = fig.add_subplot(224)
        ax4.plot(t, ax_array, label='x-방향')
        ax4.plot(t, ay_array, label='y-방향')
        ax4.set_title('시간-가속도 그래프')
        ax4.set_xlabel('시간 (s)')
        ax4.set_ylabel('가속도 (m/s²)')
        ax4.legend()
        ax4.grid(True)
        
        # 이전 그래프 제거
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # 새 그래프 표시
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 데이터 테이블 생성
        self.create_data_table(t, x, y, vx, vy, ax_array, ay_array)

        # 최고점을 그래프에 표시
        max_height_idx = np.argmax(y)
        ax1.plot(x[max_height_idx], y[max_height_idx], 'ro', label='최고점')
        ax1.legend()
        
        # 도달 거리 표시
        ground_indices = np.where(y <= 0)[0]
        if len(ground_indices) > 0:
            range_idx = ground_indices[0]
            if range_idx > 0:
                ax1.plot(x[range_idx], 0, 'go', label='도달점')
                ax1.legend()

if __name__ == "__main__":
    root = tk.Tk()
    app = ProjectileMotionCalculator(root)
    root.mainloop()
