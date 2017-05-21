clc
clear all
close all
modeset=input('Which run(0,1,2,3) do you want to run?')
testmode=modeset  % 0 is for debug mode to check if original image is processed correctly 1 is  calculates all the temporal coefficients and spatial modes 2 is for calculate specific data at specific values 
n2=1000; %Number of frames to use
fps=130000; %frame rate
scalefactor=1.473/76;
%% reading the images

 pdir=dir('*bars*.jpg');
 
 BG=imread('ref.jpg');
 initialf=imread(pdir(1).name);
 [frow fcolumn]=size(initialf);
 U=zeros(frow,fcolumn,n2);
 
if testmode==0
    lbound=36;  %Lowerbound threshold for measuring film length
    upbound=300; %Uppderbound threshold for measuring film length
    imageselect=100;
    I=imread(pdir(imageselect).name);
    Z=imsubtract(BG,I);
    regime2=(Z>=lbound)&(Z<=upbound);
    regime3=imfill(regime2,'holes');
    
    figure(1)
    imshow(I);

    figure(2)
    imshow(Z);
    
    figure(3)
    imhist(Z);
    
    figure(4)
    imshow(regime2);
    
    figure(5)
    imshow(regime3);
    
    save('imagesettings','lbound','upbound')
   
%%test mode 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% This mode calculates all eigenvalues, temporal coefficients and modes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif testmode==1
    load('imagesettings');
    imagemode=input('Used cropped image(y/n)?');
    Usum=zeros(frow,fcolumn);
    for jj1=1:n2
       
    imagename=imread(pdir(jj1).name);
    f(:,:,jj1)=im2double(imsubtract(BG,imagename));
     Usum=Usum+f(:,:,jj1);
    end
    U0=Usum./n2;
    for jj=1:(n2)
    
    %figure(1),
    %imshow(f);
    
    
    if imagemode=='n'
        U(:,:,jj) = f(:,:,jj)-U0;
    else
        regime2=(f>=lbound)&(f<=upbound);
        regime3=imfill(regime2,'holes');
        U(:,:,jj)=regime3;
    end
    
    end
    %% Mean
    % UMean = mean(U,3);
    % for i = 1:n2
    % U(:,:,i) = U(:,:,i) - UMean;
    % end

    %% Corelation matrix
    C = zeros(n2,n2); % init
    for i = 1:n2
        for j = i:n2
            C(i,j) = sum(sum(U(:,:,i).*U(:,:,j)));
            C(j,i) = C(i,j); % use symmetry
        end       
    end   
    C = C ./ (n2+1);

    %% POD modes
    [V,D] = eig(C);
    [D,I] = sort(diag(D),'descend');
    V = V(:,I);
    h=(frow);
    w=(fcolumn);
    
    podmode = zeros(h,w,n2); % init
    for ii = 1:n2
        for jj = 1:n2
            podmode(:,:,ii) = podmode(:,:,ii) + V(jj,ii) * U(:,:,jj);
            % vPOD(:,:,i) = vPOD(:,:,i) + V(j,i)*v(:,:,j);
        end
        %% Normalize
        modeFactor = 1 ./ sqrt(n2*D(ii));
        podmode(:,:,ii) = podmode(:,:,ii) * modeFactor;
    end
    
    %% Figures

    close all

    % Plot eigenvalues

    m1=figure;
    loglog([1:n2],D./sum(D),'ko-');
    % axis([1 nt 1.e-3 1])

    xlabel('Eigenvalue index','fontname','Times New Roman','fontsize',18);
    ylabel('Normalized eigenvalues','fontname','Times New Roman','fontsize',18);
    grid on
    saveas(m1,'normalized_eigenvalue_alt.tif')
   

    
    E = zeros(n2,1);
    E(1)=D(1);
    for i = 2:n2
        E(i) = E(i-1)+D(i);
        Ecomp(i)=sum(D)-E(i-1);
    end
    modal_energy=E./sum(D);
    
    m2=figure; 
    scatter(linspace(2,n2,n2),modal_energy,'k');
    grid on
    xlabel('Number of modes,r','fontname','Times New Roman','fontsize',18);
    ylabel('Cumulative modal energy, \alpha_r','fontname','Times New Roman','fontsize',18);
    saveas(m2,'modalenergy_alt.tif')
    
    m3=figure;
    truncation_error=Ecomp./sum(D);
    semilogx([1:n2],truncation_error,'k^-')
    grid on
    xlabel('Number of modes,r','fontname','Times New Roman','fontsize',18);
    ylabel('\epsilon_r','fontname','Times New Roman','fontsize',18);
    saveas(m3,'truncationerror_alt.tif')
    
    %% Modal Coefficients
    a=zeros(n2,n2);%zeros(n2,N);

    for ii3=1:n2%N    
        a(:,ii3) = sqrt(n2*D(ii3))*V(:,ii3);
    end
    
    save('POD_array_alternate','podmode','a','modal_energy','truncation_error','D','U0');

%% testmode 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% This test mode calculates many of the salient features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif testmode==2
    
    load('POD_array_alternate')
    %podmode=mode;
    timecount=(1:n2)./fps;
    
     % Selects which modes to look at
    nModes = input('range of modes: ');
     mkdir('POM_alternate')
     mkdir('PSD_alternate')
     mkdir('wave_spectrum_alternate')
    
    for ii = nModes(1):nModes(end)
        brightvalue=.1*max(max(podmode(:,:,ii)));
        darkvalue=.1*min(min(podmode(:,:,ii)));
        brightlocations=zeros(frow,1);
        darklocations=zeros(frow,1);
        % Finds all the dark and bright spots in POM
        for icol=1:fcolumn
            podslice=podmode(:,icol,ii);
            [wavemax,wavemaxindex, wavemin,waveminindex]=extrema(podslice);
            
            if length(wavemax)>0
                waveindex=[wavemaxindex' waveminindex'];
                waveindex=sort(waveindex);
                doubleamp=abs(diff(podslice(waveindex)));
                halfwaveindex=find(max(doubleamp)==doubleamp);
                halfwave=waveindex(halfwaveindex(1)+1)-waveindex(halfwaveindex(1));  
                wavelength(icol)=2*scalefactor*halfwave;
                amplitude(icol)=max(doubleamp);
            end
        end
            wl_index=find(amplitude==max(amplitude));
            wave_length(ii)=wavelength(wl_index(1));
            wavelength_avg(ii)=mean(wavelength);
          
            
            [pxx freq]=pwelch(a(:,ii),[],[],[],fps);
            peaklocation=find(pxx==max(pxx));
            wave_frequency(ii)=freq(peaklocation(1));
            
            wavex=scalefactor.*(0:fcolumn-1);
            wavey=scalefactor.*(0:frow-1);
            
            
            
            dSFx=1/fcolumn/scalefactor;
            dSFy=1/frow/scalefactor;
            SFx=(-.5/scalefactor:dSFx:.5/scalefactor-dSFx);
            SFy=(-.5/scalefactor:dSFy:.5/scalefactor-dSFy);
            kx=SFx.*2*pi;
            ky=SFy.*2*pi;
            
            wavefront=fft2(podmode(:,:,ii));
            wavefront=fftshift(wavefront);
            wavefront=abs(wavefront).^2;
            wavefront_clipped=wavefront(frow/2+1:end,fcolumn/2+1:end);
            SFx_clipped=SFx(fcolumn/2+1:end);
            SFy_clipped=SFy(frow/2+1:end);
            [ZMAX,IMAX,XMIN,IMIN] = extrema2(wavefront_clipped);
            [wrow wcol]=ind2sub(size(wavefront_clipped),IMAX);
            SF=(SFx_clipped(wcol).^2+SFy_clipped(wrow).^2).^.5;
            WAVE=1./SF;
            peak_wavelength(:,ii)=WAVE(1:3);
            
            
            %m101=figure
            %surf(kx(fcolumn/2-10:end),ky(frow/2-10:end),wavefront(frow/2-10:end,fcolumn/2-10:end))
            %colormap jet
            %xlabel('k_x','fontsize',18,'fontname','times new roman')
            %ylabel('k_y','fontsize',18,'fontname','times new roman')
            %xlim([0 2]);
            %ylim([0 2]);
            
            
            [Xq Yq]=meshgrid(0:.1:3.5);
            Vq= interp2(SFx_clipped,SFy_clipped,wavefront_clipped,Xq,Yq,'cubic');
            Kxq=Xq.*2*pi;
            Kyq=Yq.*2*pi;
            
            m100=figure
            surf(Kxq,Kyq,Vq)
            colormap(autumn(40))
            xlabel('k_x(mm^{-1})','fontsize',18,'fontname','times new roman')
            ylabel('k_y(mm^{-1})','fontsize',18,'fontname','times new roman')
            zlim([0 1.1*max(max(Vq))])
            xlim([0 Kxq(end)])
            ylim([0 Kyq(end)])
            saveas(m100,sprintf('wave_spectrum_alternate/wave_pom%1.0f.tif',ii))
            
            
            
         
        
        m1=figure;  %Plots the podmodes
        imshow(podmode(:,:,ii).*1000);
        % plot
        % [curlU,cav]= curl(X,Y,,vPOD(:,:,i));
        % pcolor(X,Y,curlU); shading interp
        caxis([-13.2 13.2]); colorbar;
        hold on;
        pomtitle=sprintf('\phi_%d',ii)
        title(['\phi_{' num2str(ii) '}'],'fontname','times new roman','fontsize',18)
        saveas(m1,sprintf('POM_alternate/pom%1.0f.tif',ii))
        
        m2=figure; %Plots the psd for given podmodes
        semilogy(freq./1000,pxx,'k','linewidth',2)
        xlabel('Frequency(kHz)','fontname','times new roman','fontsize',14)
        ylabel('PSD','fontname','times new roman','fontsize',14)
        set(gca,'XMinorTick','on','YMinorTick','on')
        xlim([0 60])
        ylim([0 100])
        saveas(m2,sprintf('PSD_alternate/psd_mode%1.0f.tif',ii))
  
    end
    
    Ncoeff=nModes;
    figure;
    plot(timecount,a(:,Ncoeff(1):Ncoeff(end)));

    
    couplemode=input('Which pairs of modes to look at? Give exact array')
    mkdir('CPSD_alternate')
    mkdir('Phase_alternate')
    
    
    
    for ii=1:length(couplemode)-1
    
        [Pxy F]=cpsd(a(:,couplemode(ii)),a(:,couplemode(ii+1)),[],[],[],fps);
        
    
        cpsdamp=abs(Pxy);
        phase=radtodeg(atan2(-imag(Pxy),real(Pxy)));
        peakcpsdfreq_index=find(max(cpsdamp)==cpsdamp);
        peakcpsd_freq=F(peakcpsdfreq_index);
        peakcpsd_phase=phase(peakcpsdfreq_index(1));
        
       
        
        orthogonal_modes(:,couplemode(ii))=[couplemode(ii);couplemode(ii+1);peakcpsd_freq;peakcpsd_phase];

    
        m3=figure;    
        plot(F./1000,cpsdamp,'k','linewidth',2);    
        ylabel('CPSD','fontname','times new roman','fontsize',14);    
        xlabel('Frequency(kHz)','fontname','times new roman','fontsize',14);    
        set(gca,'XMinorTick','on','YMinorTick','on')
        xlim([0 60])
        saveas(m3,sprintf('CPSD_alternate/cpsd_coupledmodes%d_%d.tif',couplemode(ii),couplemode(ii+1)))
    
    
        m4=figure;
        plot(F./1000,phase,'k','linewidth',2);    
        xlabel('Frequency(kHz)','fontname','times new roman','fontsize',14)
        ylabel('Phase(Degrees)','fontname','times new roman','fontsize',14)  
        set(gca,'ytick',-180:45:180,'XMinorTick','on','YMinorTick','on')
        xlim([0 60])  
        saveas(m4,sprintf('Phase_alternate/phase_coupled%d_%d.tif',couplemode(ii),couplemode(ii+1)))
    end
    
     orthogonal_modes
    
    snapshot_index=input('Which snapshots to look at?')
    isothreshold=input('Input threshold (0-100) for iso contour')
    isothreshold=isothreshold/100;
    mkdir('Snapshots_alternate')
    
    for ii=1:length(snapshot_index)
        
        snapshot=imread(pdir(snapshot_index(ii)).name);
         snapshot2=imsubtract(BG,snapshot);
        
        m5=figure;
        imagesc(snapshot2)
        %colormap(flipud(colormap));
        set(gca,'visible','off')
        saveas(m5,sprintf('Snapshots_alternate/snapshot%d.tif',snapshot_index(ii)))
        
        for jj=2:length(couplemode)
            
            if jj==2
                m100=figure; 
                snapmode=a(snapshot_index(ii),couplemode(1)).*podmode(:,:,couplemode(1));    
                snapmode=(snapmode).*1000;%+im2double(BG);            
                %superpos=(snapmode)+(snapshot);    
                imshow(snapmode)
                saveas(m100,sprintf('Snapshots_alternate/snapshot%d_mode%d.tif',snapshot_index(ii),couplemode(1)))
                
                m101=figure;
                imagesc(snapmode)
                 set(gca,'visible','off')
                saveas(m101,sprintf('Snapshots_alternate/colorsnapshot%d_mode%d.pdf',snapshot_index(ii),couplemode(1)))
                minSM=min(min(snapmode));
                maxSM=max(max(snapmode));
                sprintf('POM %d has a maximum intensity of %d and a minimum intensity of %d',couplemode(1),maxSM,minSM)
            end
                
    
            m6=figure;   
            snapmode=a(snapshot_index(ii),couplemode(jj)).*podmode(:,:,couplemode(jj));    
            snapmode=(snapmode).*1000;
            %superpos=(snapmode)+(snapshot);    
            imshow(snapmode)
            saveas(m6,sprintf('Snapshots_alternate/snapshot%d_mode%d.tif',snapshot_index(ii),couplemode(jj)))
            
            couplesnapmode=a(snapshot_index(ii),couplemode(jj-1)).*podmode(:,:,couplemode(jj-1))+a(snapshot_index(ii),couplemode(jj)).*podmode(:,:,couplemode(jj));;
            minCSM=min(min(couplesnapmode));
            maxCSM=max(max(couplesnapmode));
            
            m7=figure;
            isocolor=[204 43 245]./255;
            isocolor2=[43 164 245]./255;
            imshow(snapshot)
            hold on
            contour(couplesnapmode,[isothreshold(2).*minCSM:isothreshold(1)*minCSM],'--','linewidth',1,'linecolor',isocolor);
            contour(couplesnapmode,[isothreshold(1)*maxCSM:isothreshold(2)*maxCSM],'-','linewidth',1,'linecolor',isocolor2);
            saveas(m7,sprintf('Snapshots_alternate/snapshot%d_contour%d_%d.tif',snapshot_index(ii),couplemode(jj-1),couplemode(jj)))
            
             m8=figure;
                imagesc(snapmode.*1000)
                 set(gca,'visible','off')
                saveas(m8,sprintf('Snapshots_alternate/colorsnapshot%d_mode%d.pdf',snapshot_index(ii),couplemode(jj)))
                minSM=min(min(snapmode));
                maxSM=max(max(snapmode));
                sprintf('POM %d has a maximum intensity of %d and a minimum intensity of %d',couplemode(jj),maxSM,minSM)
        end
    end
            
    
    save('POD_array_alternate','wave_length','wave_frequency','peak_wavelength','-append')
%% Test mode 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% This test mode reconstructs the image for a selected snapshot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
elseif testmode==3
    load('POD_array_alternate')
    sayit='Image Reconstruction'
    mkdir('Image_Reconstruction_alt')
    
    snapshot_index=input('Which snapshot to reconstruct image for?')
    snapshot=imread(pdir(snapshot_index).name);
     snapshot=imsubtract(BG,snapshot);
    cumulative_snapmode=zeros(frow,fcolumn,n2);
    cumulative_snapmode(:,:,1)=a(snapshot_index,1).*podmode(:,:,1)+U0;
    for ii=2:n2
        snapmode=a(snapshot_index,ii).*podmode(:,:,ii);
        cumulative_snapmode(:,:,ii)=snapmode+cumulative_snapmode(:,:,ii-1);
    end
    
    
    m1=figure;
     imagesc(snapshot)   
     %colormap(flipud(colormap));
     set(gca,'visible','off')
     saveas(m1,sprintf('Image_Reconstruction_alt/IRsnapshot%d_original.tif',snapshot_index))
     
     m22=figure;
     imagesc(U0)
     set(gca,'visible','off')
     saveas(m22,sprintf('Image_Reconstruction_alt/IRsnapshot%d_mode0.tif',snapshot_index))
     
     m2=figure;
     imagesc(cumulative_snapmode(:,:,1))
     %colormap(flipud(colormap));
     set(gca,'visible','off')
     saveas(m2,sprintf('Image_Reconstruction_alt/IRsnapshot%d_modes%d.tif',snapshot_index,1))
     
      m3=figure;
     imagesc(cumulative_snapmode(:,:,10))
     %colormap(flipud(colormap));
     set(gca,'visible','off')
     saveas(m3,sprintf('Image_Reconstruction_alt/IRsnapshot%d_modes%d.tif',snapshot_index,10))
     
      m4=figure;
     imagesc(cumulative_snapmode(:,:,20))
     %colormap(flipud(colormap));
     set(gca,'visible','off')
     saveas(m4,sprintf('Image_Reconstruction_alt/IRsnapshot%d_modes%d.tif',snapshot_index,20))
     
      m5=figure;
     imagesc(cumulative_snapmode(:,:,50))
     %colormap(flipud(colormap));
     set(gca,'visible','off')
     saveas(m5,sprintf('Image_Reconstruction_alt/IRsnapshot%d_modes%d.tif',snapshot_index,50))
     
      m6=figure;
     imagesc(cumulative_snapmode(:,:,100))
     %colormap(flipud(colormap));
     set(gca,'visible','off')
     saveas(m6,sprintf('Image_Reconstruction_alt/IRsnapshot%d_modes%d.tif',snapshot_index,100))
     
      m7=figure;
     imagesc(cumulative_snapmode(:,:,500))
     %colormap(flipud(colormap));
     set(gca,'visible','off')
     saveas(m7,sprintf('Image_Reconstruction_alt/IRsnapshot%d_modes%d.tif',snapshot_index,500))
     
       m8=figure;
     imagesc(cumulative_snapmode(:,:,1000))
     %colormap(flipud(colormap));
     set(gca,'visible','off')
     saveas(m8,sprintf('Image_Reconstruction_alt/IRsnapshot%d_modes%d.tif',snapshot_index,1000))
    

end
