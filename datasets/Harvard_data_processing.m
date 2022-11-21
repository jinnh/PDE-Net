clear 
clc
close all

dataset = 'Hararvd';
upscale = 4;
mode = {'train', 'test'};

train = ''; % all train file name
test = '';  % all test file name

image_size = 256;

savePath = ['./HSI/Harvard/TrainTestMat_256/',num2str(upscale)]; %save test set  to "savePath"
if ~exist(savePath, 'dir')
    mkdir(savePath)
end

%% obtian all the original hyperspectral image
for d = 1:2

    % download the Hararvd dataset and split the dataset into train_data and test_data
    if strcmp(mode(d), 'train')
        srPath = './HSI/Harvard/Train';
        disp('-----deal with: training data'); 
    else
        srPath = './HSI/Harvard/Test';
        disp('-----deal with: testing data'); 
    end  
    
    srFile=fullfile(srPath);
    srdirOutput=dir(fullfile(srFile));
    srfileNames={srdirOutput.name}';
    number = length(srfileNames);

    for index = 1 : number
        name = char(srfileNames(index));
        if(isequal(name,'.')||... % remove the two hidden folders that come with the system
               isequal(name,'..'))
                   continue;
        end
        disp(['-----deal with:',num2str(index),'----name:',name]); 
        load([srPath,'/',name])
        data =ref;
        clear lbl
        clear ref

        %% normalization
        data = data/(1.0*max(max(max(data))));
        if strcmp(mode(d), 'train')
            data = data(8:1032, 56:1336,:); % Training data            
           %% obtian HR and LR hyperspectral image    
            for index1 = 1 : 4
                for index2 = 1 : 5
                  %% obtian HR and LR hyperspectral image    
                    train_data = data(1+(image_size*(index1-1)):image_size*index1, image_size*(index2-1)+1:image_size*(index2-1)+image_size,:);
                    file_name = ['train_', name(1:length(name)-4), '_', int2str(i), '.mat'];
                    img = reshape(train_data, size(train_data,1)*size(train_data,2), 31);
                    HR = modcrop(train_data, upscale);
                    LR = imresize(HR,1/upscale,'bicubic'); %LR  
                    save([savePath,'/',file_name], 'HR', 'LR')
                    train = strvcat(char(train), char(file_name));
                    i=i+1;
                    clear HR
                    clear LR
                end
            end
        else
            data = data(1:512, 1:512,:); % Testing data
           %% obtian HR and LR hyperspectral image    
            img = reshape(data, size(data,1)*size(data,2), 31);
            HR = modcrop(data, upscale);
            LR = imresize(HR,1/upscale,'bicubic'); %LR  
            
            file_name = ['test_', name];
            save([savePath,'/',file_name], 'HR', 'LR')
            test = strvcat(char(test), char(file_name));
            clear HR
            clear LR
        end    

    end
end

%% save training and testing filename
save('./HSI/Harvard/harvard_train_test_filename.mat', 'train', 'test')
