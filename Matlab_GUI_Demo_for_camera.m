clear
clc

imaqhwinfo                                            % 获取硬件设备信息

obj = videoinput('winvideo', 1 , 'YUY2_2560x720');    % 此处数字: 1表示相机的设备号，是由系统分配给相机.该设备号不总是1，需要根据实际情况来设置 !!!!!!
set(obj, 'FramesPerTrigger', 1);                      % 设置捕获属性
set(obj, 'TriggerRepeat', Inf);

%定义一个监控界面
hf = figure('Units', 'Normalized', 'Menubar', 'None','NumberTitle', 'off', 'Name', '莱娜视觉--机器视觉--技术专家');
ha = axes('Parent', hf, 'Units', 'Normalized', 'Position', [0.05 0.2 0.85 0.7]);
axis off

%定义两个按钮控件
hb1 = uicontrol('Parent', hf, 'Units', 'Normalized','Position', [0.25 0.05 0.2 0.1], 'String', '打开相机', 'Callback', ['objRes = get(obj, ''VideoResolution'');' ...
     'nBands = get(obj, ''NumberOfBands'');' ...
     'hImage = image(zeros(objRes(2), objRes(1), nBands));' ...
     'preview(obj, hImage);']);
hb2 = uicontrol('Parent', hf, 'Units', 'Normalized','Position', [0.55 0.05 0.2 0.1], 'String', '双目抓拍', 'Callback', 'imwrite(ycbcr2rgb(getsnapshot(obj)), ''image.bmp'', ''bmp'')');
