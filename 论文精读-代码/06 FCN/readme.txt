--------------------------
tensorflow�汾
--------------------------

�ٷ���ַ��https://github.com/shekkizh/FCN.tensorflow

�ο���ַ��
1��https://blog.csdn.net/m0_37407756/article/details/83379026����ϸ��������ô�ӹٷ����뵽ѵ���Լ������ݼ���
2��https://blog.csdn.net/MOU_IT/article/details/81073149������һƪ��࣬�ѹٷ����븴����һ�飩

----------------------------------------------------------------------------------
ͨ��batch_json_to_dataset.py��batch_color_map.py��3to1_channel.py����ͼ��Ĵ���
batch_json_to_dataset.py: ���Բο�����1�������ṩ�������ļ�������ļ���Ҫ��������json�ļ�����labelme��עͼ��õ���json��ת��Ϊpngͼ��ת���꿴�����Ǻں�����һƬ����
python batch_json_to_dataset.py json(���json���ļ���) gt_png3(outputλ��)

batch_color_map.py: ������ת��Ϊ�ں�����ͼƬ������ɫ��ʹ���ܹ�������ע��ͼƬ����ɫ��ͼ������ͨ���ģ�������ЩͼƬ����ֱ��ʹ�ã�

3to1_channel.py�����������ɵĲ�ɫ��ͨ��ͼƬ����ת��������ת��Ϊ1ͨ����

----------------------------------------------------------------------------------
gen_pickle.py�������洦��õ��ļ������ΪData_zoo/MIT_ScenceParsing/�ٷ�����ĸ�ʽ��Ȼ����������ļ����ɶ�Ӧ��pickle�ļ���
�ļ��и�ʽΪ��
./Data_zoo/MIT_ScenceParsing|
                             |ADEChallengeData2016|
                                                   |annotations|
                                                                |training
                                                                |validation
                                                   |images|
                                                           |training
                                                           |validation
                             |MITSceneParsing.pickle��������Ҫ�Ǵ���ļ���·����

----------------------------------------------------------------------------------

FCN.py: NUM_OF_CLASSES��Ϊ�����+1(������)��