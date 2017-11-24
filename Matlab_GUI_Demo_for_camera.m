clear
clc

imaqhwinfo                                            % ��ȡӲ���豸��Ϣ

obj = videoinput('winvideo', 1 , 'YUY2_2560x720');    % �˴�����: 1��ʾ������豸�ţ�����ϵͳ��������.���豸�Ų�����1����Ҫ����ʵ����������� !!!!!!
set(obj, 'FramesPerTrigger', 1);                      % ���ò�������
set(obj, 'TriggerRepeat', Inf);

%����һ����ؽ���
hf = figure('Units', 'Normalized', 'Menubar', 'None','NumberTitle', 'off', 'Name', '�����Ӿ�--�����Ӿ�--����ר��');
ha = axes('Parent', hf, 'Units', 'Normalized', 'Position', [0.05 0.2 0.85 0.7]);
axis off

%����������ť�ؼ�
hb1 = uicontrol('Parent', hf, 'Units', 'Normalized','Position', [0.25 0.05 0.2 0.1], 'String', '�����', 'Callback', ['objRes = get(obj, ''VideoResolution'');' ...
     'nBands = get(obj, ''NumberOfBands'');' ...
     'hImage = image(zeros(objRes(2), objRes(1), nBands));' ...
     'preview(obj, hImage);']);
hb2 = uicontrol('Parent', hf, 'Units', 'Normalized','Position', [0.55 0.05 0.2 0.1], 'String', '˫Ŀץ��', 'Callback', 'imwrite(ycbcr2rgb(getsnapshot(obj)), ''image.bmp'', ''bmp'')');
