function Plot_SingleUnit_Example(Info,UnitName)

      hf = figure;
      set(hf,'Position',[100 100 800 800]);
      
      %******** Note, can use StimRast (all trials) or 
      %********    StimRast2 (eliminated trials with prior saccade, 350 ms
      subplot('Position',[0.1 0.35 0.35 0.60]);
      if (1)
          Rast = Info.SacImage.StimRast2;  % two columns, spike time x trial
      else
          Rast = Info.SacImage.StimRast;
      end
      plot(Rast(:,1),Rast(:,2),'k.','Markersize',1); hold on;
      axis tight;
      V = axis;
      plot([0,0],[V(3),V(4)],'k-');
      zz = find( Info.SacImage.StimUU == max(Info.SacImage.StimUU));
      PeakLatency = (1000*Info.SacImage.StimTT(zz(1)));
      title(sprintf('%s Latency: %3.1f ms',UnitName,PeakLatency));
      % xlabel('Time (ms)');
      ylabel('Saccade Event');
      subplot('Position',[0.1 0.10 0.35 0.20]);
      aa = [Info.SacImage.StimTT fliplr(Info.SacImage.StimTT)];
      bb = [(Info.SacImage.StimUU+(2*Info.SacImage.StimSU)),...
            fliplr(Info.SacImage.StimUU-(2*Info.SacImage.StimSU))];
      fill(aa,bb,[0.5,0.5,0.5],'Linestyle','None'); hold on;
      plot(Info.SacImage.StimTT,Info.SacImage.StimUU,'k-','LineWidth',2);
      axis tight;
      V = axis;
      axis([V(1) V(2) 0 V(4)]);
      Rb = mean(Info.SacImage.StimUU(find(Info.SacImage.StimTT<0)));
      plot([V(1),V(2)],[Rb,Rb],'k:');
      plot([0,0],[V(3),V(4)],'k-');
      
      xlabel('Time (ms)');
      %********
      
      %****** Temporal Response Kernel
      subplot('Position',[0.6 0.80 0.23 0.15]);
      PPDelay = -8;  % monitor delay of ProPixx at 240hz
      Info.Hart.tXX = Info.Hart.tXX + PPDelay;
      %**** correct offset in timing from monitor delay
      aa = [Info.Hart.tXX fliplr(Info.Hart.tXX)];
      bb = [(Info.Hart.mcounts+(2*Info.Hart.msem)),...
            fliplr(Info.Hart.mcounts-(2*Info.Hart.msem))];
      fill(aa,bb,[0.7,0.7,0.7],'Linestyle','None'); hold on;
      plot(Info.Hart.tXX,Info.Hart.mcounts,'k-','LineWidth',1);
      axis tight;
      V = axis;
      plot([0,0],[V(3),V(4)],'k-');
      plot([V(1),V(2)],[Info.Hart.ymean,Info.Hart.ymean],'k:');
      xlabel('Time (ms)');
      ylabel('Rate (sp/s)');
      title('Temporal Response');
      
      %****** Joint Tuning
      subplot('Position',[0.6 0.525 0.30 0.20]);
      ymean = Info.Hart.ymean;
      colormap('gray');
      imagesc(Info.Hart.SpatOris,log2(Info.Hart.SpatFrqs),Info.Hart.uu',[ymean (1.2*max(max(Info.Hart.uu)))]);
      colorbar;
      text(250,4,'Rate (sp/s)','Rotation',90,'Fontsize',10);
      xlabel('Orientation (degs)');
      ylabel('Log2 SF.(cyc/deg)');
      
      %****** Orientation Tuning
      subplot('Position',[0.6 0.300 0.23 0.15]);
      nn = length(Info.Hart.SpatOris);
      zz = 1:nn;
      aa = [Info.Hart.SpatOris fliplr(Info.Hart.SpatOris)];
      bb = [(Info.Hart.otune(zz)+(2*Info.Hart.sotune(zz))) ...
            fliplr(Info.Hart.otune(zz)-(2*Info.Hart.sotune(zz)))];
      fill(aa,bb,[0.7,0.7,0.7],'Linestyle','None'); hold on;
      plot(Info.Hart.SpatOris,Info.Hart.otune(zz),'k-','LineWidth',1);
      axis tight;
      V = axis;
      plot([V(1),V(2)],[Info.Hart.ymean,Info.Hart.ymean],'k:');
      xlabel('Orientation (degs)');
      ylabel('Rate (sp/s)');
      
      %****** SF Tuning
      subplot('Position',[0.6 0.100 0.23 0.15]);
      aa = [log2(Info.Hart.SpatFrqs) fliplr(log2(Info.Hart.SpatFrqs))];
      bb = [(Info.Hart.stune+(2*Info.Hart.sstune)) ...
            fliplr(Info.Hart.stune-(2*Info.Hart.sstune))];
      fill(aa,bb,[0.7,0.7,0.7],'Linestyle','None'); hold on;
      plot(log2(Info.Hart.SpatFrqs),Info.Hart.stune,'k-','LineWidth',1);
      axis tight;
      V = axis;
      plot([V(1),V(2)],[Info.Hart.ymean,Info.Hart.ymean],'k:');
      xlabel('Log2 SF (cyc/deg)');
      ylabel('Rate (sp/s)');
      
      
return;