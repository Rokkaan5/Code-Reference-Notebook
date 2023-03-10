students <- c("Sean","Louisa","Frank","Farhad","Li")
midterm <- c(80,90,93,82,95)

students

final <- c(78,84,95,82,91)

course_grades <- 0.4*midterm+0.6*final # final course grades

#-----
final>midterm
 
(final<midterm) & (midterm>80)