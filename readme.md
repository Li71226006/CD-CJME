## CD-CJME


<img src=".\img\\logo.png" alt="tiny" style="zoom:80%;" />


# Cable-Driven Continuum Joint Motion Estimation (CD-CJME) Dataset

## 1.Experimental Prototype

The **Cable-Driven Continuum Joint Motion Estimation (CD-CJME)** dataset is based on a flexible surgical robot developed by the School of Mechanical Engineering, Hefei University of Technology, which serves as the experimental platform. This robot features three degrees of freedom (**DOF**), including rotation, translation, and bending, making it suitable for research on continuum joints. The figure below illustrates the appearance and structure of the experimental prototype, providing insight into the design and layout of the device.

<img src=".\img\\Experimental prototype.png" alt="tiny" style="zoom:40%;" />

## 2.Data Collection Setup

The **CD-CJME** dataset is designed for the **Flexible Cable-Driven System**, capturing sensor parameters under two conditions: unloaded and loaded. A total of 12 data sets were collected across these scenarios, detailed as follows:

- **Unloaded condition**:  
  A composite sinusoidal signal was input to the motor to drive the motion of the continuum joint, and 2 data sets were collected.  

- **Loaded condition**:  
  Different weights (5g, 10g, 15g) were suspended at the end of the continuum joint to simulate various force conditions. For each weight, the same composite sinusoidal signal was used to collect 2, 4, and 4 data sets, respectively.  

In total, 10 data sets were collected under loaded conditions and 2 under unloaded conditions. Each time series contains **5000 time points** with a sampling rate of **50 Hz** and a duration of **100 seconds**.  

## 3.Dataset Variables

The dataset includes 16 variables representing the states of the  **Flexible Cable-Driven System**:

<img src=".\img\\system.png" alt="tiny" style="zoom:40%;" />

1. **Three forces (N)** and **three torques (N·M)** measured by a six-dimensional force sensor.  
2. The **X-coordinates (mm)** and **Y-coordinates (mm)** of three marked points (**Marked Point1**, **Marked Point2**, and **Marked Point3**).  
3. The **actual bending angle (degrees)** of the continuum joint, derived from **Marked Point1** and **Marked Point4** using a two-dimensional visual detection system.  
4. The **tension (N)** of two flexible tendons.  
5. The **theoretical bending angle (degrees)**.  

Below are the state characteristics of each feature under the unloaded condition:

<img src=".\img\\Data_feature.png" alt="large" style="zoom:200%;" />

## 4.Contact

For more information and access to datasets please contact us ([wangzhengyu_hfut@hfut.edu.cn](mailto:wangzhengyu_hfut@hfut.edu.cn)，xinzhou.xu@njupt.edu.cn, or lzq_hfut_2021@163.com)
