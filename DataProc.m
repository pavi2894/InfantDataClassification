

load('dataset_unlabeled.mat')
names = {'Kotimittaus_VAURAS+33','Kotimittaus_VAURAS+34','Kotimittaus_VAURAS+35','Kotimittaus_VAURAS34','Kotimittaus_VAURAS35','Kotimittaus_VAURAS39','Kotimittaus_VAURAS41_kaksoset','Kotimittaus_VAURAS43','Kotimittaus_VAURAS46','Kotimittaus_VAURAS47','Kotimittaus_VAURAS51','Kotimittaus_VAURAS58','Kotimittaus_VAURAS61','Kotimittaus_VAURAS72','Kotimittaus_VAURAS73','Kotimittaus_VAURAS77','Kotimittaus_VAURAS78','Kotimittaus_VAURAS80','Kotimittaus_VAURAS81','Kotimittaus_VV54','Kotimittaus_VV55','Kotimittaus_VV61','Kotimittaus_VV62','Kotimittaus_VV63','Kotimittaus_VV_xx','Kotimittaus_pilot2','Kotimittaus_vaihe2_VAURAS82','Kotimittaus_vaihe2_VV64','Kotimittaus_vaihe2_VV66'}
folder  = 'unlabeled_DATA/';

for i = 1:length(names)
      name = names{i};
      %name = convertCharsToStrings(name)
      mkdir([folder, name]);
      
      acc_data = L(i).acc_data;
      file = strcat(folder,name,"/acc_data.mat");
      save(file,'acc_data') ;
      
      file = strcat(folder,name,"/gyro_data.mat");
      gyro_data = L(i).gyro_data;
      save(file,'gyro_data');     
end



load('dataset_full_v2.mat')  
LABELLED names = {'Kotimittaus_VAURAS35','Kotimittaus_VAURAS38','Kotimittaus_VAURAS39','Kotimittaus_VAURAS41_kaksoset','Kotimittaus_VAURAS42_kaksoset','Kotimittaus_VAURAS43','Kotimittaus_VAURAS46','Kotimittaus_VAURAS47','Kotimittaus_VAURAS51','Kotimittaus_VAURAS52','Kotimittaus_VAURAS53','Kotimittaus_VAURAS63','Kotimittaus_VV54','Kotimittaus_VV55','Kotimittaus_VV_xx','Kotimittaus_pilot1','Kotimittaus_pilot2','baby10','baby11','baby12','baby13','baby14','baby15','baby16','baby17','baby18','baby19','baby20','baby21','baby22','baby23','baby24','baby25','baby26','baby3','baby4','baby5','baby6','baby7','baby8','baby9'};
folder  = 'labeled_DATA/';
for i = 1:length(names)
      name = names{i};
      %name = convertCharsToStrings(name)
      mkdir([folder, name]);
      
      acc_data = L(i).acc_data;
      file = strcat(folder,name,"/acc_data.mat");
      save(file,'acc_data') ;
      
      file = strcat(folder,name,"/gyro_data.mat");
      gyro_data = L(i).gyro_data;
      save(file,'gyro_data');
      
       file = strcat(folder,name,"/timestamp.mat");
       timestamp = L(i).timestamp;
       save(file,'timestamp');
       
       file = strcat(folder,name,"/posture_oh.mat");
       posture_oh = L(i).posture_oh;
       save(file,'posture_oh');
       
       file = strcat(folder,name,"/movement_oh.mat");
       movement_oh = L(i).movement_oh;
       save(file,'movement_oh');
       
       file = strcat(folder,name,"/mask.mat");
       mask = L(i).mask;
       save(file,'mask');
end
