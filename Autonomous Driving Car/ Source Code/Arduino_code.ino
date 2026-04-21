#include <Servo.h>
#include <math.h>

Servo myServo;

#define Servopin 6
#define EnA 5
#define En1 4
#define En2 3

unsigned int pre_angle = 50, servoAngle = 50, servo_attack = 50;
float kp = 0.10;
unsigned long time, lasttime = 0;

void setup() {
  Serial.begin(115200);
  pinMode(EnA, OUTPUT);
  pinMode(En1, OUTPUT);
  pinMode(En2, OUTPUT);
  myServo.attach(Servopin);
  myServo.write(servoAngle);
}

void Forward(int speed) {
  analogWrite(EnA, speed);
  digitalWrite(En1, LOW);
  digitalWrite(En2, HIGH);
}

void servo_rotate(unsigned int servoAngle) {
  servo_attack = round(pre_angle + kp * (servoAngle - pre_angle));
  pre_angle = servo_attack;
  myServo.write(servo_attack);
}

void loop() {
  time = millis();

  if (Serial.available()) {
    servoAngle = Serial.read()-40;

    Forward(90);
    lasttime = time;
  }

  servo_rotate(servoAngle);

  if (time - lasttime > 3000) {
    if(servoAngle!=50){
      servoAngle = 50;
      myServo.write(50);
    }
    Forward(0);
  }
}
