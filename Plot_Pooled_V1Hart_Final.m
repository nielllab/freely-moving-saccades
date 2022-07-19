function Plot_Pooled_V1Hart_Onset()
%*** loads the Preload file created by the Pooled routine
%*** The following data structs should load in the Preload file
%ISACMOD = [];  % saccade modulation psths for images (each row a neuron)
                % time steps are 1ms, with 50 being saccade offset
%GSACMOD = [];  % saccade modulation psths for gratings (each row a neuron)
%SFTUNE = [];   % spatial freq tuning curves (each row a neuron)
%TFTUNE = [];   % temporal freq tuning (fft of temporal kernel), each row a neuron
%ORTUNE = [];   % orientation tuning per neuron
%BASEMU = [];   % baseline firing rate, to natural images (inclusion criteria?)
%PEAKIM = [];   % time of the peak latency of nat image modulation
%SFGRID = [];   % for each neuron, the spatial frequencies uses in RF 
%ANIMID = [];   % which of the two animals, 1=Allen, 2=Logan


%**** need metrics to summarize tuning, low or high spatial/temp freq pref
% we could compute them later in the plotting functions instead ...
% SFI = [];  % spatial frequency index, computed from spat freq tuning curve
% TFI = [];  % temporal frequency index, computed from temp freq tuning curve

Environment_Preload = 'Pooled_V1Hart_Preload_Final.mat';
Environment_Totload = 'Pooled_V1Hart_TotalInfo.mat';
if ~exist(Environment_Preload)
    disp(sprintf('Failed in loading precomputed lists from %s ...',Environment_Preload));
    return;
else
    load(Environment_Preload)
    disp('Loading completed');
end
%********** load a huge cell arrays, TotalInfo, with complete info to plot
%********** out aspects of single unit tuning
ExampleUnits = [39,57,74,76,68];  % will plot out indivudal examples selected
%**** comments on units looked at going step by step
% Unit 1 - late peak, strong supp, higher SF tuning
% Unit 4 - early peak, low rate
% Unit 7 - early peak, low SF tuning (weak ori), low rate
% Unit 8 - later intermediate (100ms), suppression, higher SF tuning, low rate (iso)
% Unit 9 - late peak, higher SF tuning, typical modulation
% Unit 10, early peak, lower SF tuning (high rate)
% Unit 18, early peak, low rate, low SF tuning
% Unit 21, intermediate peak, intermediate tuning.
% Unit 30, late peak, strong supp, higher SF tuning
% Unit 49, late peak, strong supp, higher SF tuning
% Unit 39, early peak, high rate, low SF tuning
% Unit 57, intermediate peak, intermediate tuning
% Unit 59, late peak, higher SF tuning
% Unit 68, Simple Cell?  Weak orientation tuning, high SF
% Unit 74, late intermediate, strong tuning, high SF
% Unit 76, very late, strong tuning, higher SF
% Unit 95, late, strong tuning, higher SF
% Unit 96, same thing (low rate)
if ~exist(Environment_Totload)
    disp(sprintf('Failed in loading total information from %s ...',Environment_Totload));
    return;
else
    load(Environment_Totload)
    disp('Loading completed');
end
%***** step through an plot single example units
for k = 1:length(TotalInfo)
   if (1) 
    if ismember(k,ExampleUnits)  % plot the examples
       fname = sprintf('Unit %d',k);
       Plot_SingleUnit_Example(TotalInfo{k},fname);
    end
   end
   %******* or uncomment here to step through one by one
%    if (0)
%        fname = sprintf('Unit %d',k);
%        Plot_SingleUnit_Example(TotalInfo{k},fname);
%        input('check');
%        close all;
%    end
end

%********* isolate analyses to one animal?
ANIMAL = 0;  %1 is Allen only, 2 is Logan only, 0 is both
if (ANIMAL)
    azz = find(ANIMID == ANIMAL);
else
    azz = (1:length(ANIMID))';
end
%********
if (1)
    BASEMU = BASEMU(azz);
    BASEMU2 = BASEMU2(azz);
    %*******  swap in ISACMOD2 for ISACMOD, excluded 350ms
    if (1)
        tmp = ISACMOD(azz,:);
        ISACMOD = ISACMOD2(azz,:);
        ISACMOD2 = tmp;
        tmp = PEAKIM(azz);
        PEAKIM = PEAKIM2(azz);
        PEAKIM2 = tmp;
    else
        ISACMOD = ISACMOD(azz,:);
        ISACMOD2 = ISACMOD2(azz,:);
        PEAKIM = PEAKIM(azz);
        PEAKIM2 = PEAKIM2(azz);
    end
    %****
    GSACMOD = GSACMOD(azz,:);
    SFTUNE = SFTUNE(azz,:);
    TFTUNE = TFTUNE(azz,:); 
    ORTUNE = ORTUNE(azz,:);
    SFGRID = SFGRID(azz,:);
    ANIMID = ANIMID(azz);
    disp(sprintf('Isolating analysis to animal %d',ANIMAL));    
end
%****** Adjust for the time axes
XTIME = mean(TSACMOD)*1000;  % should all be the same (in ms)
YNERS = 1:size(GSACMOD,1);
SPATF = [1,2,4,8,16];
TEMPF = (60/16)*(0:8); 
%***********************************

%******** next go into plotting and analysis routines
%***** first one, just show raw results
if (1)
     M = length(PEAKIM);
     rr = sortrows([(1:M)' PEAKIM],2);
     rord = rr(:,1);
     ranger = [0 2.5];
     comap = [];
     for cm = 1:100
         val = (cm/100);
         if (cm <=50)
            comap = [comap ; [2*val,2*val,1]];
         else
            comap = [comap ; [1,2*(1-val),2*(1-val)]];             
         end
     end
     %**********
     hf = figure;
     set(hf,'Position',[100 100 1200 800]);
     subplot('Position',[0.025 0.1 0.15 0.8]);
     imagesc(XTIME,YNERS,log2(1+ISACMOD2(rord,:)),ranger); hold on;  % original
     colormap(comap);
     plot([0,0],[0,M],'k-');
     xlabel('Time');
     ylabel('Cells');
     title(sprintf('SacMod(Allen %d Logan %d)',sum(ANIMID==1),sum(ANIMID==2)));
     %****
     subplot('Position',[0.200 0.1 0.15 0.8]);
     imagesc(XTIME,YNERS,log2(1+ISACMOD(rord,:)),ranger); hold on;  %excluded 350ms
     colormap(comap);
     plot([0,0],[0,M],'k-');
     xlabel('Time');
     ylabel('Cells');
     title(sprintf('EX SacMod(Allen %d Logan %d)',sum(ANIMID==1),sum(ANIMID==2)));
     %****
     subplot('Position',[0.400 0.1 0.15 0.8]);
     imagesc(XTIME,YNERS,log2(1+GSACMOD(rord,:)),ranger); hold on;
     colormap(comap);
     plot([0,0],[0,M],'k-');
     xlabel('Time');
     ylabel('Cells');
     title(sprintf('Grating(Allen %d Logan %d)',sum(ANIMID==1),sum(ANIMID==2)));
     subplot('Position',[0.625 0.1 0.15 0.8]);
     imagesc(SPATF,YNERS,log2(1+SFTUNE(rord,:)),ranger);
     xlabel('Sp. Freq');
     ylabel('Cells');
     title('SF Tune');
     subplot('Position',[0.825 0.1 0.15 0.8]);
     imagesc(TEMPF,YNERS,log2(1+TFTUNE(rord,:)),ranger);
     xlabel('Temp Freq');
     ylabel('Cells');
     title('TF Tune');
     %*******
end

%****** now, let's try to run through all cells and compute stats on tuning
OFI = [];  % orientation tuning ... may want to separate non-tuned units
SFI = [];  % spatial frequency index, computed from spat freq tuning curve
TFI = [];  % temporal frequency index, computed from temp freq tuning curve
SPF = [];  % spatial frequency pref
TPF = [];  % temporal freq pref
%********
for fr = 1:size(SFTUNE,1)
    otune = ORTUNE(fr,:);
    stune = SFTUNE(fr,:);
    ttune = TFTUNE(fr,:);
    %********
    ofi = nanstd(otune)/nanmean(otune);
    sfi = nanstd(stune)/nanmean(stune);
    tfi = nanstd(ttune)/nanmean(ttune);
    %********
    spatf = SPATF; % load exact values (if differ exp or animal)
    tempf = TEMPF;
    %***
    svec = max((stune-1),0).^2;
    if (sum(svec) == 0)
        svec = (stune-min(stune)) .^ 2;  % if response suppressed from baseline
    end
    spref = nansum(svec .* spatf)/nansum(svec); 
    tvec = max((ttune-1),0).^2;
    tpref = nansum(tvec .* tempf)/nansum(tvec); 
    %********
    OFI = [OFI ; ofi];
    SFI = [SFI ; sfi];
    TFI = [TFI ; tfi];
    SPF = [SPF ; spref];
    TPF = [TPF ; tpref];
end
%***** seperate cluster of non-selective units (center/surround)
FFI = sqrt( OFI.^2 + SFI.^2); 
TUNETHRESH = 0.15;
zmain = find( FFI >= TUNETHRESH);
zflat = find( FFI < TUNETHRESH);
%*******

if (1)
    %****** plot a scatter of population and parameters
    hf = figure;
    set(hf,'Position',[200 200 1500 500]);
    subplot('Position',[0.1 0.1 0.25 0.8]);
    plot(OFI(zflat),SFI(zflat),'ro','MarkerSize',3); hold on;
    plot(OFI(zmain),SFI(zmain),'ko','MarkerSize',3); hold on;
    xlabel('Ori tune');
    ylabel('SF tune');  
    %*****
    subplot('Position',[0.4 0.1 0.25 0.8]);
    plot(OFI(zflat),TPF(zflat),'ro','MarkerSize',3); hold on;
    plot(OFI(zmain),TPF(zmain),'ko','MarkerSize',3); hold on;
    xlabel('Ori tune');
    ylabel('TF pref');      
    %******
    subplot('Position',[0.7 0.1 0.25 0.8]);
    plot(OFI(zflat),SPF(zflat),'ro','MarkerSize',3); hold on;
    plot(OFI(zmain),SPF(zmain),'ko','MarkerSize',3); hold on;
    xlabel('Ori tune');
    ylabel('SF pref');             
end
%*****

%****** Redo the analyses but show units with no ori or sf tuning separate
%*******
if (1)
   hf = figure;
   set(hf,'Position',[100 100 1200 800]);
   for k = 1:2
     if (k == 2)
         zz = zmain;
         vb = 0.50;
     else
         zz = zflat;
         vb = 0.25;
     end
     YNERS = 1:length(zz);
     vv = 0.05+((k-1)*0.35);
     M2 = length(zz);
     rr = sortrows([(1:M2)' PEAKIM(zz)],2);
     rord = rr(:,1);
     %******
     subplot('Position',[0.025 vv 0.2 vb]);
     imagesc(XTIME,YNERS,log2(1+ISACMOD(zz(rord),:)),ranger); hold on;
     colormap(comap);
     plot([0,0],[0,M],'k-');
     xlabel('Saccade Onset (ms)');
     ylabel('Cells');
     if (k==2)
        title(sprintf('Selective(Allen %d Logan %d)',sum(ANIMID(zz)==1),sum(ANIMID(zz)==2)));
     else
        title(sprintf('Non-Select(Allen %d Logan %d)',sum(ANIMID(zz)==1),sum(ANIMID(zz)==2)));
     end
     subplot('Position',[0.275 vv 0.2 vb]);
     imagesc(XTIME,YNERS,log2(1+GSACMOD(zz(rord),:)),ranger); hold on;
     colormap(comap);
     plot([0,0],[0,M],'k-');
     xlabel('Saccade Onset (ms)');
     ylabel('Cells');
     title(sprintf('SacMod Grating'));
     subplot('Position',[0.525 vv 0.2 vb]);
     imagesc(SPATF,YNERS,log2(1+SFTUNE(zz(rord),:)),ranger);
     xlabel('Sp. Freq');
     ylabel('Cells');
     title('SF Tune');
     subplot('Position',[0.775 vv 0.2 vb]);
     imagesc(TEMPF,YNERS,log2(1+TFTUNE(zz(rord),:)),ranger);
     xlabel('Temp Freq');
     ylabel('Cells');
     title('TF Tune');
     %*******
   end
end

%******** among the selective units, plot correlation latency peak and tune
if (1)
    hf = figure;
    set(hf,'Position',[400 300 1500 500]);
    subplot('Position',[0.1 0.15 0.25 0.7]);
    z1 = find(ANIMID(zmain)==1);
    z2 = find(ANIMID(zmain)==2);
    gcolo = [0.6,0.6,0.6];
    bcolo = [0,0,0.6];
    plot(PEAKIM(zmain(z1)),SPF(zmain(z1)),'ko','Markersize',3,'Color',gcolo); hold on;
    plot(PEAKIM(zmain(z2)),SPF(zmain(z2)),'bo','Markersize',3,'Color',bcolo,'Linewidth',2); hold on;    
    [r,p] = corr(PEAKIM(zmain),SPF(zmain),'type','Spearman');
    if (ANIMAL == 0)
      [r1,p1] = corr(PEAKIM(zmain(z1)),SPF(zmain(z1)),'type','Spearman');
      [r2,p2] = corr(PEAKIM(zmain(z2)),SPF(zmain(z2)),'type','Spearman');
      title(sprintf('R=%5.3f(%5.3f,%5.3f) p=%6.4f(%6.4f,%6.4f)',r,r1,r2,p,p1,p2));
    else
      title(sprintf('R=%5.3f p=%6.4f',r,p));    
    end
    xlabel('Peak latency(ms)');
    ylabel('SF pref (cyc/deg)');
     %***
    subplot('Position',[0.4 0.15 0.25 0.7]);
    plot(PEAKIM(zmain(z1)),TPF(zmain(z1)),'ko','Markersize',3,'Color',gcolo); hold on;
    plot(PEAKIM(zmain(z2)),TPF(zmain(z2)),'bo','Markersize',3,'Color',bcolo,'Linewidth',2); hold on;
    [r,p] = corr(PEAKIM(zmain),TPF(zmain),'type','Spearman');
    if (ANIMAL == 0)
      [r1,p1] = corr(PEAKIM(zmain(z1)),TPF(zmain(z1)),'type','Spearman');
      [r2,p2] = corr(PEAKIM(zmain(z2)),TPF(zmain(z2)),'type','Spearman');
      title(sprintf('R=%5.3f(%5.3f,%5.3f) p=%6.4f(%6.4f,%6.4f)',r,r1,r2,p,p1,p2));
    else
      title(sprintf('R=%5.3f p=%6.4f',r,p));    
    end
    xlabel('Peak latency(ms)');
    ylabel('TF pref (hz)');
    %***
    subplot('Position',[0.7 0.15 0.25 0.7]);
    plot(SPF(zmain(z1)),TPF(zmain(z1)),'ko','MarkerSize',3,'Color',gcolo); hold on;
    plot(SPF(zmain(z2)),TPF(zmain(z2)),'bo','MarkerSize',3,'Color',bcolo,'Linewidth',2); hold on;
    [r,p] = corr(SPF(zmain),TPF(zmain),'type','Spearman');
    xlabel('SF pref (cyc/deg)');
    ylabel('TF pref (hz)');
    title(sprintf('Magno vs Parvo: R=%5.3f(p=%6.4f)',r,p));
end

if (0) % show saccade modulation by image and grating
   hf = figure;
   set(hf,'Position',[100 100 800 400]);
   for k = 1:2
     if (k==1)
       subplot('Position',[0.1 0.15 0.35 0.7]);
       zz = find( ANIMID(zmain) == 1);
       titlename = sprintf('Allen N=%d',length(zz));
     else
       subplot('Position',[0.6 0.15 0.35 0.7]);
       zz = find( ANIMID(zmain) == 2);
       titlename = sprintf('Logan N=%d',length(zz));
     end
     Iuu = nanmean(ISACMOD(zmain(zz),:));
     Isu = nanstd(ISACMOD(zmain(zz),:))/sqrt(length(zmain(zz)));
     Guu = nanmean(GSACMOD(zmain(zz),:));
     Gsu = nanstd(GSACMOD(zmain(zz),:))/sqrt(length(zmain(zz)));
     xx = -50:250;
     aa = [xx fliplr(xx)];
     bb = [(Iuu+(2*Isu)) fliplr(Iuu-(2*Isu))];
     fill(aa,bb,[0,0,0],'FaceAlpha',0.3,'LineStyle','none'); hold on;
     plot(xx,Iuu,'k-','LineWidth',2); hold on;
     bb = [(Guu+(2*Gsu)) fliplr(Guu-(2*Gsu))];
     fill(aa,bb,[1,0,1],'FaceAlpha',0.3,'LineStyle','none'); hold on;
     plot(xx,Guu,'m-','LineWidth',2);
     plot([0,0],[0,5],'k-');
     plot([-50,200],[1,1],'k-');
     axis([-50 200 0 5]);
     xlabel('Saccade Onset (ms)');
     ylabel('Saccade Modulation'); 
     title(titlename);
   end
end

if (0) % show saccade modulation by image and grating
   hf = figure;
   set(hf,'Position',[100 100 800 400]);
   for k = 1:2
     if (k==1)
       subplot('Position',[0.1 0.15 0.35 0.7]);
       zz = find( ANIMID(zflat) == 1);
       titlename = sprintf('Allen N=%d',length(zz));
     else
       subplot('Position',[0.6 0.15 0.35 0.7]);
       zz = find( ANIMID(zflat) == 2);
       titlename = sprintf('Logan N=%d',length(zz));
     end
     Iuu = nanmean(ISACMOD(zflat(zz),:));
     Isu = nanstd(ISACMOD(zflat(zz),:))/sqrt(length(zflat(zz)));
     Guu = nanmean(GSACMOD(zflat(zz),:));
     Gsu = nanstd(GSACMOD(zflat(zz),:))/sqrt(length(zflat(zz)));
     xx = -50:250;
     aa = [xx fliplr(xx)];
     bb = [(Iuu+(2*Isu)) fliplr(Iuu-(2*Isu))];
     fill(aa,bb,[0,0,0],'FaceAlpha',0.3,'LineStyle','none'); hold on;
     plot(xx,Iuu,'k-','LineWidth',2); hold on;
     bb = [(Guu+(2*Gsu)) fliplr(Guu-(2*Gsu))];
     fill(aa,bb,[1,0,1],'FaceAlpha',0.3,'LineStyle','none'); hold on;
     plot(xx,Guu,'m-','LineWidth',2);
     plot([0,0],[0,5],'k-');
     plot([-50,200],[1,1],'k-');
     axis([-50 200 0 5]);
     xlabel('Saccade Onset (ms)');
     ylabel('Saccade Modulation'); 
     title(titlename);
   end
end

if (1)
   %********* PCA analyses on saccade modulation, show groups by PC
   %******** the k-means clustering
   KM = 4;
   colo = [[0.8,0.6,0];[0.8,0.4,0];[0.6,0,0.6];[0.3,0.3,0.7]]; %;[0.3,0.8,0.3]];
   sacmod = ISACMOD(zmain,:);
   for k = 1:size(sacmod,1)
       zsacmod(k,:) = (sacmod(k,:)-1)/(max(sacmod(k,:))-min(sacmod(k,:)));
   end
   [COEFF, SCORE, LATENT] = pca(zsacmod);
   TPC1 = -ISACMOD * COEFF(:,1);   % projection to PC1
   
   PC1 = SCORE(:,1);
   PC2 = SCORE(:,2);
   PC3 = SCORE(:,3);
   IDX = kmeans(zsacmod,KM);
   lats = [];
   for k = 1:KM
     id = find(IDX == k);
     uu = nanmean(sacmod(id,:));
     lats = [lats ; find(uu == max(uu))];  
   end
   soto = sortrows([(1:KM)' lats],2);
   ksoto = soto(:,1);
   %******
   hf = figure;
   set(hf,'Position',[400 200 1200 400]);
   subplot('Position',[0.1 0.15 0.25 0.7]);
   for k = 1:KM
     id = find(IDX == ksoto(k));
     plot(PC1(id),PC2(id),'ko','Color',colo(k,:),'Markersize',3); hold on;
   end
   xlabel('PC1');
   ylabel('PC2');
   title('PCA on Saccade Modulation');
   subplot('Position',[0.4 0.15 0.25 0.7]);
   for k = 1:KM
     id = find(IDX == ksoto(k));
     plot(PC2(id),PC3(id),'ko','Color',colo(k,:),'Markersize',3); hold on;
   end
   xlabel('PC2');
   ylabel('PC3');
   title('PCA on Saccade Modulation');
   %******
   subplot('Position',[0.7 0.15 0.25 0.7]);
   xx = XTIME; 
   for k = 1:KM
     id = find(IDX == ksoto(k));
     uu = nanmean(sacmod(id,:));
     su = nanstd(sacmod(id,:))/sqrt(length(id));
     aa = [xx fliplr(xx)];
     bb = [(uu+(2*su)) fliplr(uu-(2*su))];
     fill(aa,bb,colo(k,:),'FaceAlpha',0.3,'LineStyle','none'); hold on;
     plot(xx,uu,'k-','LineWidth',2,'Color',colo(k,:)); hold on;
   end 
   plot([0,0],[0,7],'k-');
   plot([-50,200],[1,1],'k-');
   axis([-50 200 0 7]);
   xlabel('Saccade Onset (ms)');
   ylabel('Saccade Modulation');
   title(sprintf('N = %d,%d,%d,%d,%d',sum(IDX==ksoto(1)),sum(IDX==ksoto(2)),...
                  sum(IDX==ksoto(3)),sum(IDX==ksoto(4))));
end

%******** among the selective units, plot correlation PC1 and tune
if (0)
    hf = figure;
    set(hf,'Position',[400 300 1500 500]);
    subplot('Position',[0.1 0.15 0.25 0.7]);
    z1 = find(ANIMID(zmain)==1);
    z2 = find(ANIMID(zmain)==2);
    plot(TPC1(zmain(z1)),SPF(zmain(z1)),'ko','Markersize',3); hold on;
    plot(TPC1(zmain(z2)),SPF(zmain(z2)),'bo','Markersize',3); hold on;
    [r,p] = corr(TPC1(zmain),SPF(zmain),'type','Spearman');
    if (ANIMAL == 0)
      [r1,p1] = corr(TPC1(zmain(z1)),SPF(zmain(z1)),'type','Spearman');
      [r2,p2] = corr(TPC1(zmain(z2)),SPF(zmain(z2)),'type','Spearman');
      title(sprintf('R=%5.3f(%5.3f,%5.3f) p=%6.4f(%6.4f,%6.4f)',r,r1,r2,p,p1,p2));
    else
      title(sprintf('R=%5.3f p=%6.4f',r,p));    
    end
    xlabel('PC1');
    ylabel('SF pref (cyc/deg)');
     %***
    subplot('Position',[0.4 0.15 0.25 0.7]);
    plot(TPC1(zmain(z1)),TPF(zmain(z1)),'ko','Markersize',3); hold on;
    plot(TPC1(zmain(z2)),TPF(zmain(z2)),'bo','Markersize',3); hold on;
    [r,p] = corr(TPC1(zmain),TPF(zmain),'type','Spearman');
    if (ANIMAL == 0)
      [r1,p1] = corr(TPC1(zmain(z1)),TPF(zmain(z1)),'type','Spearman');
      [r2,p2] = corr(TPC1(zmain(z2)),TPF(zmain(z2)),'type','Spearman');
      title(sprintf('R=%5.3f(%5.3f,%5.3f) p=%6.4f(%6.4f,%6.4f)',r,r1,r2,p,p1,p2));
    else
      title(sprintf('R=%5.3f p=%6.4f',r,p));    
    end
    xlabel('PC1');
    ylabel('TF pref (hz)');
end


return;    