% Programa exemplo para formação de visão em Matlab / python set 2019
% Miguel Oliveira
clear all % limpar memoria toda
close all % fechar todas as janelas
clc % limpar o ecra do command window

% para ler a imagem do disco
A = imread('images/imagem_teste.jpeg');

% para mostrar a imagem numa janela
%imshow(A);

% converter a imagem para grayscale
B = rgb2gray(A);

% figure % criar nova figura
% imshow(B) % mostar imagem B
% title('Imagem B')
% pause(0.1)

C = B > 200;
M = logical(C);

% figure
% imshow(M)
% title('Imagem M ')
% pause(0.1)

% Proposta do Alex
linha_corte = size(B,1)/2;

% pintar de preto os pixeis de cima da imagem
M2 = M; % copiar M para M2
M2(1:linha_corte, :) = 0; % Pintar parte de cima de M2 a preto

%
% figure
% imshow(M2)
% title('Imagem M2 -> apagar parte cima')
% pause(0.1)


% solução João (ficar apenas com a parte de baixo da imagem)
M3 = M(linha_corte:end, :); % copiar parte de baixo de M para M3


% figure
% imshow(M4)
% title('Imagem M3 -> selecionar parte baixo')
% pause(0.1)

% Miguel 
figure; imshow(M3)
M4 = M3;

[L, num] = bwlabel(M3);

figure; imshow(L)
title('L')

figure(3);
for i=1:num
   
    M = L == i;
    
%     npb = nnz(M);
    npb = sum(sum(M)); % calcular num pixeis brancsos
    
    if npb > 800 % encontrei o carro
       M4 = M4 - M; 
    end
    
    
    subplot(1,2,1)
    pause(0.2)
    imshow(M)
    pause(0.2)
    title(['Objecto ' num2str(i) ' npb=' num2str(npb)])
    
    
    subplot(1,2,2)
    imshow(M4)
    
    pause(0.5)
    
end

figure; imshow(L)

%% Luis 2

% figure; imshow(M3)
% F = 1/11/11*ones(11,11); % criar filtro
% R = filter2(F,M3); % aplicar filtro à imagem
% 
% whos
% figure; imshow(R)

%% Bruno 
% 
% figure; imshow(M3)
% 
% F = 1/11/11*ones(11,11); % criar filtro
% R = filter2(F,M3); % aplicar filtro à imagem
% 
% whos
% figure; imshow(R)
% 
% 
% M = R < 6/11/11 & R > 2/11/11;
% 
% figure; imshow(M); 
% title('mascara')
% 




%% Lucas 
% figure; imshow(M3)
% % M4dir = logical(M3 * 0);
% % M4dir = logical(zeros(size(M3,1), size(M3,2)))
% M4dir = false(size(M3,1), size(M3,2));
% M4esq = M4dir;
% 
% for lin = 1: size(M3,1) % percorrer as linhas
%     
%     for col = 1:size(M3,2) % percorrer as colunas esq para dir
%         if M3(lin, col) == 1
%             M4esq(lin, col) = 1;    
%             break
%         end
%         
%     end
%         
%     for col = size(M3,2):-1:1 % percorrer as colunas dir para esq
%         if M3(lin, col) == 1
%             M4dir(lin, col) = 1;    
%             break
%         end
%         
%     end
% end
% 
% figure; imshow(M4esq)
% figure; imshow(M4dir)
%     
% 
% Mmerge = or(M4esq, M4dir);
% figure; imshow(Mmerge);
% 

%% Luís
% percorrer cada linha da imagem e se encontrar mais do que T
% pixeis brancos seguidos apagá-los

% 
% T = 10;
% D = uint8(M3)*255;
% figure
% imshow(D)
% title('Imagem M3 -> selecionar parte baixo')
% pause(0.1)
% 
% 
% for linha = 1:size(M3,1)
%     % linha = 60;
%     
%     brancos_seguidos = 0;
%     cols_brancos = [];
%     nos_brancos = 0; % sinaliza se estou a percorrer uma mancha brancos
%     for col = 1:size(M3,2)
%         
%         if M3(linha, col) == 1 % se o pixel é branco
%             if nos_brancos == 1
%                 brancos_seguidos = brancos_seguidos + 1;
%                 cols_brancos = [cols_brancos col];
%             else % encontro branco mas não estava nos _brancos
%                 nos_brancos = 1;
%                 brancos_seguidos = brancos_seguidos + 1;
%                 cols_brancos = [cols_brancos col];
%             end
% %             D(linha,col) = 230;
% %             imshow(D)
% %             pause(0.01)
%         else % encontra pixel preto
%             if nos_brancos == 1
%                 % teste do Luis
%                 if brancos_seguidos > T % mancha branca muito grande apagar
%                     D(linha, cols_brancos) = 0;
%                 end
%                 brancos_seguidos = 0;
%                 cols_brancos = [];
%                 
%                 
%             else % encontro preto mas não estava nos_brancos
%                 % nothing to do for now ...
%             end
%         end
%         
%         
%     end
%     
% end
% 
% imshow(D)