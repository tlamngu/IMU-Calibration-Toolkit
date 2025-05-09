# IMU Calibrator Toolkit

_Version 1.1.0 By Versys Research Team_  
**Developer:** Zeaky Nguyen

## 🇻🇳 Tiếng Việt

### **Giới thiệu**

Bộ công cụ này hỗ trợ **cân bằng cảm biến IMU** và **ước tính phương hướng** bằng nhiều phương pháp tiên tiến. Nó cung cấp các tính năng giúp cải thiện độ chính xác và hiệu suất của hệ thống AHRS (**EKF** / **Madgwick**).

### **Các tính năng chính**

-   **Cân bằng cảm biến gia tốc và con quay hồi chuyển:** Thu thập dữ liệu tĩnh để ước lượng độ lệch (bias) của cảm biến. Cân bằng accelerometer giúp cô lập tác động của trọng lực.
-   **Cân bằng cảm biến từ trường:** Phép khớp elip (**Least Squares**) trên dữ liệu magnetometer thô để xác định độ lệch hard-iron và biến dạng soft-iron (hệ số tỷ lệ và độ nhạy liên trục).
-   **Ước tính phương hướng:** Lựa chọn thuật toán AHRS (**EKF hoặc Madgwick**) để hợp nhất dữ liệu từ **accelerometer**, **gyroscope**, và **magnetometer**, cung cấp ước tính phương hướng 3D dựa trên quaternion. EKF cũng ước lượng độ lệch gyroscope theo thời gian thực.
-   **Điều chỉnh R-Value cho EKF:** Công cụ hỗ trợ ước lượng phương sai nhiễu đo của **accelerometer** và **magnetometer** (giá trị R cho EKF) từ dữ liệu tĩnh.
-   **Điều chỉnh Beta cho Madgwick:** Công cụ đề xuất giá trị `beta` cho bộ lọc **Madgwick** dựa trên đặc tính nhiễu của gyroscope từ dữ liệu tĩnh.
-   **Hiển thị dữ liệu:** Trực quan hóa dữ liệu cảm biến theo thời gian thực và hiển thị phương hướng 3D.

### **Ứng dụng**

Bộ công cụ này hữu ích trong nhiều lĩnh vực như **robot tự hành**, **hệ thống định vị**, **điều khiển chuyển động**, và các ứng dụng yêu cầu độ chính xác cao của IMU.

----------

## 🇬🇧 English

### **Introduction**

This toolkit provides **IMU calibration** and **orientation estimation** using advanced techniques. It enhances accuracy and performance for **AHRS algorithms** (**EKF / Madgwick**).

### **Key Features**

-   **Accelerometer & Gyroscope Calibration:** Static data collection to estimate **sensor offsets** (biases). Accelerometer calibration helps **isolate gravity effects**.
-   **Magnetometer Calibration:** **Ellipsoid fitting (Least Squares)** applied to raw magnetometer data to determine **hard-iron offsets** and **soft-iron distortions** (scale factors and cross-axis sensitivities).
-   **Orientation Estimation:** Selectable **AHRS algorithms** (**EKF or Madgwick**) fuse data from **accelerometer**, **gyroscope**, and **magnetometer**, providing a **robust 3D orientation estimate** using quaternions. The **EKF also estimates gyroscope biases online**.
-   **EKF R-Value Tuning:** A utility to estimate **accelerometer and magnetometer measurement noise variances** (**R values for EKF**) from static data.
-   **Madgwick Beta Tuning:** A utility that suggests an appropriate `beta` gain for the **Madgwick filter**, based on **gyroscope noise characteristics** from static data.
-   **Data Visualization:** Real-time plotting of sensor data and **3D orientation display**.

### **Applications**

This toolkit is valuable for fields such as **autonomous robotics**, **navigation systems**, **motion control**, and applications requiring **high precision and stability** from IMUs.

----------