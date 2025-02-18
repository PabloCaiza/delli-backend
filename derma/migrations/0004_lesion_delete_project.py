# Generated by Django 4.2.3 on 2023-07-16 03:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('derma', '0003_project_my_boolean'),
    ]

    operations = [
        migrations.CreateModel(
            name='Lesion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(null=True, upload_to='')),
                ('edad', models.IntegerField()),
                ('temperatura', models.DecimalField(decimal_places=2, max_digits=10)),
                ('dolor_cabeza', models.BooleanField(default=False)),
                ('conjuntivitis', models.BooleanField(default=False)),
                ('malestar_general', models.BooleanField(default=False)),
                ('ganglios_hinchados', models.BooleanField(default=False)),
                ('tos', models.BooleanField(default=False)),
                ('moqueo', models.BooleanField(default=False)),
                ('dolor_garganta', models.BooleanField(default=False)),
                ('diarrea', models.BooleanField(default=False)),
                ('vomito', models.BooleanField(default=False)),
                ('nauseas', models.BooleanField(default=False)),
                ('comenzon', models.BooleanField(default=False)),
                ('perdida_apetito', models.BooleanField(default=False)),
                ('dolor_tragar', models.BooleanField(default=False)),
                ('hinchazon', models.BooleanField(default=False)),
                ('hinchazon_boca', models.BooleanField(default=False)),
                ('dolor_abdominal', models.BooleanField(default=False)),
                ('escalofrio', models.BooleanField(default=False)),
                ('perdida_gusto', models.BooleanField(default=False)),
                ('dolor_dentadura', models.BooleanField(default=False)),
                ('cara', models.BooleanField(default=False)),
                ('torso', models.BooleanField(default=False)),
                ('cabeza', models.BooleanField(default=False)),
                ('extremidades_superiores', models.BooleanField(default=False)),
                ('extremidades_inferiores', models.BooleanField(default=False)),
                ('genitales', models.BooleanField(default=False)),
                ('manos', models.BooleanField(default=False)),
                ('boca', models.BooleanField(default=False)),
                ('pies', models.BooleanField(default=False)),
            ],
        ),
        migrations.DeleteModel(
            name='Project',
        ),
    ]
