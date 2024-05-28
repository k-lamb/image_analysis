library(ggplot2)
library(dplyr)
#converting px to units, calculating density and stomate size

#read in for narrow model with extras
xy_narrow <- read.csv("~/Desktop/Documents/Research/Stomata/processed/xy_final.csv", header=F, sep=",")
colnames(xy_narrow) <- c("a", "b", "c2", "id")

#convert to useful units (from px to mm)
xy_narrow$a_mm <- xy_narrow$a / 1310
xy_narrow$b_mm <- xy_narrow$b / 1310

xy_narrow$area <- xy_narrow$a_mm * xy_narrow$b_mm * pi

#make a table of counts per image
test <- as.data.frame(table(xy_narrow$id))
real_count <- read.csv("~/Desktop/Documents/Research/Stomata/wide_narrow_real_count.csv", header = T, sep = ",")

real_count$full_048 <- test$Freq
write.csv(real_count, "~/Desktop/Documents/Research/Stomata/wide_narrow_real_count.csv")

#########################################################################

#calculate linear model
ggplot(real_count, aes(x=real, y=full_050))+
  geom_point()+
  geom_smooth(method="lm")+
  theme_bw()

#get the R^2
narrow.lm <- lm(full_050 ~ real, data = real_count)
summary(narrow.lm)

#########################################################################

# does [time] (or standardized time) per [image] (or box) explain bad images? 
# answer: no [time: pval = 0.64, r2 = -0.009] (standardized time: pval = 0.11, r2 = 0.02)

time <- read.csv("~/Desktop/Documents/Research/Stomata/processed/time.csv", header=F, sep=",")
colnames(time) <- c("time_s")
real_count$time_050_2 <- time$time_s

#regress standardized time per box (per image) and delta for the image
real_count$delta  <- real_count$real - real_count$full_050_2 #get difference between machine and human
real_count$time_std <- real_count$time_050_2 / real_count$full_050_2

ggplot(real_count, aes(x=delta, y=time_std))+
  geom_point()+
  geom_smooth(method="lm")+
  theme_bw()
  
time.lm <- lm(time_std ~ delta, data=real_count)
summary(time.lm)

#########################################################################

thresh_r2 <- read.csv("~/Desktop/Documents/Research/Stomata/trouble/thresh_r2.csv", header = T, sep = ",")
n <- as.numeric(dim(thresh_r2)[1])

temp <- data.frame(spline(thresh_r2, n=n*10))
ggplot(thresh_r2, aes(x=thresh, y=r2))+
  geom_point()+
  geom_line(data=temp, aes(x=x,y=y))+
  theme_bw()







