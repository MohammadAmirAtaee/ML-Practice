# SimSiam on CIFAR-10

پروژه‌ای آموزشی و قابل‌اجرا برای یادگیری خودنظارتی با SimSiam و ارزیابی خطی روی CIFAR-10. تمام دستورهای زیر برای Windows PowerShell نوشته شده‌اند و CPU کافی است.

## ایده‌های اصلی

یادگیری خودنظارتی از خود داده، هدف آموزشی می‌سازد و به برچسب انسانی نیاز ندارد. SimSiam دو نمای تقویت‌شده از یک تصویر را به یک encoder، projector و predictor می‌دهد. خروجی دو شاخه با شباهت کسینوسی مقایسه می‌شود؛ در یکی از شاخه‌ها `stop-gradient` اعمال می‌شود تا شبکه به راه‌حل فروپاشیده نرود. پس از pretraining، encoder ثابت می‌ماند و فقط یک classifier خطی برای ۱۰ کلاس آموزش می‌بیند.

تابع زیان برابر میانگین دو negative cosine similarity است. CIFAR-10 شامل ۶۰٬۰۰۰ تصویر رنگی ۳۲×۳۲ در کلاس‌های airplane، automobile، bird، cat، deer، dog، frog، horse، ship و truck است.

## نصب

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## اجرا

```powershell
python main.py download
python main.py train-pretrain --epochs 1 --limit 256
python main.py train-linear --epochs 1 --limit 256
python main.py train-supervised --epochs 1 --limit 256
python main.py demo
```

برای آزمایش واقعی، `--limit` را حذف کنید و تعداد epoch را افزایش دهید. نمودارها با `evaluation.visualization.save_curves` ذخیره می‌شوند. معماری با ورودی `[batch,3,32,32]` کار می‌کند؛ checkpointها در `checkpoints/` قرار می‌گیرند.

## دمو

پس از `train-linear`، `python main.py demo` را اجرا کنید. تصویر upload و نمایش داده می‌شود، encoder و classifier اجرا می‌شوند، کلاس پیش‌بینی‌شده و احتمال هر ده کلاس نمایش داده می‌شود. اگر checkpoint وجود نداشته باشد، پیام واضح اجرای `python main.py train-linear` نشان داده می‌شود.

## سخت‌افزار و عیب‌یابی

CPU اجرا را ممکن می‌کند اما آموزش کامل CIFAR-10 روی CPU کند است؛ برای ارائه از subset کوچک استفاده کنید. خطای نبودن داده یعنی `python main.py download` را اجرا کنید. خطای نبودن checkpoint یعنی ابتدا pretraining و سپس linear evaluation را اجرا کنید. اگر PowerShell اجرای activation را مسدود کرد، فقط `Set-ExecutionPolicy -Scope Process Bypass` را در همان پنجره اجرا کنید.

ساختار پروژه مستقل است: `models` معماری، `datasets` داده و augmentation، `training` حلقه‌های آموزشی، `evaluation` معیار و نمودار، `utils` seed/checkpoint و `app` دمو را نگه می‌دارد.
