# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models

class DataAirdb(models.Model):
    deviceid = models.IntegerField()
    date = models.DateTimeField()
    temp = models.FloatField()
    distance = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'data_airdb'


class DataCorrossion(models.Model):
    deviceid = models.IntegerField(db_column='deviceID')  # Field name made lowercase.
    date = models.DateTimeField()
    humidity = models.FloatField()
    celcius = models.FloatField()
    busvoltage = models.FloatField()
    shuntvoltage = models.FloatField()
    loadvoltage = models.FloatField()
    current_ma = models.FloatField(db_column='current_mA')  # Field name made lowercase.
    power_mw = models.FloatField(db_column='power_mW')  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'data_corrossion'


class DataKorosi(models.Model):
    deviceid = models.IntegerField(db_column='deviceID')  # Field name made lowercase.
    date = models.DateTimeField()
    huma = models.CharField(max_length=255)
    temp = models.CharField(max_length=255)
    current = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = 'data_korosi'


class MasterSungai(models.Model):
    sungai_id = models.AutoField(primary_key=True)
    sungai_nama = models.CharField(max_length=255)
    sungai_latitude = models.CharField(max_length=24, blank=True, null=True)
    sungai_longtitude = models.CharField(max_length=24, blank=True, null=True)
    sungai_deviceid = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'master_sungai'


class MasterUser(models.Model):
    user_id = models.AutoField(primary_key=True)
    user_nama = models.CharField(max_length=255)
    user_email = models.CharField(max_length=255)
    user_username = models.CharField(max_length=255)
    user_password = models.CharField(max_length=255)
    user_level = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'master_user'
